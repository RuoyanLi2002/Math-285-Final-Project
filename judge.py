"""
Lesson-distillation pipeline for the CMixture trading agents.

Workflow
--------
1. Iterate over data_point_*.json traces in a directory and split them into
   two case sets:
     - BAD  : final_decision != 0 and sign disagrees with target return.
     - GOOD : final_decision != 0 and sign agrees with target return.
2. Embed each case (OpenAI embeddings on the OHLCV window + orchestrator
   response) and cluster within each set. Clusters smaller than
   `min_cluster_size` are dropped as noise.
3. For each surviving cluster, ask a judge LLM (GPT-5.5) to distill ONE
   generalizable lesson from a few representative cases. Bad-case lessons
   may be DIRECTIONAL ("favor X when Y") OR ABSTAIN ("output 0 when Z")
   OR CONFIRMATION ("require X before acting on Y"). Good-case lessons
   reinforce what worked.
4. Maintain a bounded lesson book (num_lessons = 5). For every new lesson
   the judge decides one of:
       SKIP       - already covered by an existing lesson
       UPDATE     - small additive edit to an existing lesson
       APPEND     - orthogonal new info; add (only if under cap)
       CONTRADICT - conflicts with an existing lesson; drop BOTH

Usage
-----
    export OPENAI_API_KEY=sk-...
    python distill_lessons.py --traces-dir ./traces --out lessons.json \
        --include-successes --min-cluster-size 3

Set --dry-run to walk the pipeline without calling the LLM (useful for
testing on a large corpus before paying for tokens).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUDGE_MODEL = "gpt-5.5"          # the judge model the user asked for
# Some newer OpenAI models (gpt-5 family, o-series) reject any non-default
# temperature. Set to None to omit the parameter entirely; set to a float
# only if you know the model accepts it.
JUDGE_TEMPERATURE: Optional[float] = None
NUM_LESSONS_DEFAULT = 5

# Hard guardrail on UPDATE actions. If the judge says UPDATE but the new
# text shares less than this fraction of content words with the lesson it
# claims to be updating, we treat the action as SKIP instead. This stops
# UPDATEs that are really stealth rewrites.
UPDATE_MIN_SIMILARITY = 0.5

# Embedding + clustering knobs. We embed each case (OHLCV window + the
# orchestrator's reasoning) with OpenAI embeddings, then group similar
# cases so we can distill ONE lesson per cluster instead of one lesson
# per trade. Lessons distilled from many similar mistakes generalize
# better than lessons fit to a single noisy data point.
EMBEDDING_MODEL = "text-embedding-3-small"
CLUSTER_MIN_SIZE_DEFAULT = 3       # drop clusters smaller than this
CLUSTER_TARGET_SIZE = 5            # used to auto-pick KMeans k
CLUSTER_REPS_PER_CLUSTER = 3       # # of cases sent to the judge per cluster
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TraceCase:
    """A single trace, tagged as a 'bad' (wrong-sign) or 'good' (right-sign)
    action. Zero-action traces are never loaded."""
    kind: str                       # "bad" or "good"
    path: str
    data_point_id: int
    final_decision: float
    target_return: float
    formatted_series: str
    analyst_outputs: List[Dict[str, Any]]
    orchestrator_response: str

    @property
    def direction_taken(self) -> str:
        return "LONG (+1)" if self.final_decision > 0 else "SHORT (-1)"

    @property
    def correct_direction(self) -> str:
        return "LONG (+1)" if self.target_return > 0 else "SHORT (-1)"

    def embed_text(self) -> str:
        """Text used to compute the embedding for clustering. We use the
        OHLCV table plus the orchestrator's reasoning — this captures both
        what the market looked like and how the system thought about it."""
        return (
            f"{self.formatted_series}\n\n"
            f"Orchestrator reasoning:\n{self.orchestrator_response.strip()}"
        )


@dataclass
class LessonBook:
    """Bounded list of lessons plus a per-event audit log."""
    num_lessons: int = NUM_LESSONS_DEFAULT
    lessons: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_lessons": self.num_lessons,
            "lessons": list(self.lessons),
            "history": list(self.history),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Step 1 — load + split into bad / good
# ---------------------------------------------------------------------------

def load_cases(traces_dir: str) -> Dict[str, List[TraceCase]]:
    """Scan every data_point_*.json in `traces_dir` and split traces into:
       - 'bad' : final_decision != 0 and sign disagrees with target_return
       - 'good': final_decision != 0 and sign agrees with target_return
    Zero / no-action traces are dropped from both sets."""
    pattern = os.path.join(traces_dir, "data_point_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matched {pattern!r}.")

    bad: List[TraceCase] = []
    good: List[TraceCase] = []
    for p in paths:
        try:
            with open(p) as f:
                trace = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"[skip] {p}: {e}", file=sys.stderr)
            continue

        final = float(trace.get("final_decision", 0.0))
        eval_blk = trace.get("eval") or {}
        target = eval_blk.get("target_return")
        if target is None or final == 0.0:
            continue
        target = float(target)

        same_sign = (final > 0 and target > 0) or (final < 0 and target < 0)
        kind = "good" if same_sign else "bad"

        case = TraceCase(
            kind=kind,
            path=p,
            data_point_id=int(trace.get("data_point_id", -1)),
            final_decision=final,
            target_return=target,
            formatted_series=trace["input"]["formatted_series"],
            analyst_outputs=trace["analysts"]["outputs"],
            orchestrator_response=trace["orchestrator"]["raw_response"],
        )
        (good if same_sign else bad).append(case)

    return {"bad": bad, "good": good}


# ---------------------------------------------------------------------------
# Step 2 — judge LLM wrapper
# ---------------------------------------------------------------------------

class JudgeLLM:
    """Thin wrapper around the OpenAI chat API for the judge model."""

    def __init__(self, model: str = JUDGE_MODEL,
                 temperature: Optional[float] = JUDGE_TEMPERATURE,
                 dry_run: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.client = None
        if not dry_run:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai") from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        if self.dry_run:
            return self._fake_response(system, user)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        }
        # Omit temperature entirely for models that lock it (e.g. gpt-5 family).
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # Deterministic stub used when --dry-run is set.
    @staticmethod
    def _fake_response(system: str, user: str) -> str:
        if "Return ONLY a JSON object" in system:
            # Integration step: alternate between APPEND and SKIP for testing.
            decision = "APPEND" if "lesson_1" not in user else "SKIP"
            return json.dumps({
                "action": decision,
                "target_index": None,
                "updated_lesson": None,
                "reasoning": "[dry-run stub]",
            })
        # Distillation step.
        return ("When recent bars show a strong directional move with rising "
                "volume, do not fade it without an explicit reversal signal.")


# ---------------------------------------------------------------------------
# Step 2b — embeddings client (for clustering)
# ---------------------------------------------------------------------------

class EmbeddingClient:
    """Thin wrapper around the OpenAI embeddings API. In --dry-run mode it
    returns deterministic random vectors so the rest of the pipeline still
    works without spending tokens."""

    def __init__(self, model: str = EMBEDDING_MODEL, dry_run: bool = False,
                 batch_size: int = 64):
        self.model = model
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.client = None
        if not dry_run:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("pip install openai") from e
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")
            self.client = OpenAI(api_key=api_key)

    def embed(self, texts: List[str]):
        """Returns a (N, D) numpy array. Cosine-normalises rows for
        downstream KMeans/cosine-distance use."""
        import numpy as np

        if not texts:
            return np.zeros((0, 8), dtype=np.float32)

        if self.dry_run:
            # Deterministic pseudo-embeddings keyed on text hash so dry
            # runs are reproducible and clusterable.
            rng = np.random.default_rng(RANDOM_SEED)
            vecs = []
            for t in texts:
                local = np.random.default_rng(abs(hash(t)) % (2**32))
                vecs.append(local.standard_normal(64).astype(np.float32))
            arr = np.stack(vecs, axis=0)
        else:
            rows = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = self.client.embeddings.create(
                    model=self.model, input=batch
                )
                rows.extend(d.embedding for d in resp.data)
            arr = np.asarray(rows, dtype=np.float32)

        # L2-normalise so dot product == cosine similarity.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms


# ---------------------------------------------------------------------------
# Step 3 — clustering
# ---------------------------------------------------------------------------

def cluster_cases(cases: List[TraceCase],
                  embedder: EmbeddingClient,
                  min_cluster_size: int = CLUSTER_MIN_SIZE_DEFAULT,
                  target_cluster_size: int = CLUSTER_TARGET_SIZE,
                  ) -> List[List[TraceCase]]:
    """KMeans-cluster cases by embedding, drop clusters smaller than
    `min_cluster_size`. Returns a list of clusters, each a list of cases."""
    if not cases:
        return []
    # Too few cases to bother clustering — treat as one group if it meets
    # the minimum, otherwise drop everything.
    if len(cases) < 2 * min_cluster_size:
        return [cases] if len(cases) >= min_cluster_size else []

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "Clustering requires scikit-learn. Install with: pip install scikit-learn"
        ) from e

    embeddings = embedder.embed([c.embed_text() for c in cases])
    k = max(2, len(cases) // target_cluster_size)

    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(embeddings)
    labels = km.labels_

    buckets: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        buckets.setdefault(int(lbl), []).append(idx)

    clusters: List[List[TraceCase]] = []
    for lbl, indices in buckets.items():
        if len(indices) < min_cluster_size:
            continue
        # Order members by distance to centroid so the first few are the
        # most representative.
        import numpy as np
        centroid = embeddings[indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        sims = embeddings[indices] @ centroid
        order = np.argsort(-sims)
        clusters.append([cases[indices[i]] for i in order])

    # Sort biggest cluster first so the most evidence-backed lessons come
    # in early and shape what counts as orthogonal later.
    clusters.sort(key=len, reverse=True)
    return clusters


# ---------------------------------------------------------------------------
# Step 4 — distill ONE lesson per cluster
# ---------------------------------------------------------------------------

DISTILL_BAD_SYSTEM = """You are a trading post-mortem judge.

You will be shown a CLUSTER of similar losing trades made by a multi-agent
trading system. The cluster represents a recurring failure mode: each trade
in it has the wrong directional sign vs. the realised next-day return.

Your job is to produce ONE short, generalizable LESSON that, if the analysts
had followed it, would have made them less likely to make this kind of
mistake on FUTURE windows.

The lesson can take any of these shapes — pick whichever best fits the
common pattern across the cluster:

  DIRECTIONAL : "When <signal/condition>, favor <long|short>."
  ABSTAIN     : "When <signal/condition>, output 0 (no action) rather
                 than betting either direction."
  CONFIRMATION: "Do not take a <long|short> on <signal> unless
                 <additional confirmation> is also present."

Prefer ABSTAIN or CONFIRMATION lessons when the cluster shows the system
making over-confident bets on ambiguous setups. Prefer DIRECTIONAL only
when the data clearly points the other way.

The lesson must:
  - apply to FUTURE windows (do not name specific tickers, dates, or trades),
  - name a specific pattern or signal (no generic advice like "be careful"),
  - fit in a single sentence, at most ~30 words.

Respond with ONLY the lesson text. No preamble, no quotes, no bullet points.
"""


DISTILL_GOOD_SYSTEM = """You are a trading post-mortem judge.

You will be shown a CLUSTER of similar WINNING trades made by a multi-agent
trading system. Each trade in the cluster took a directional position whose
sign matched the realised next-day return — the system was right.

Your job is to produce ONE short, generalizable LESSON that captures the
pattern the analysts correctly identified, so they keep applying it on
FUTURE windows.

The lesson should be phrased as a positive directional rule:

  "When <signal/condition>, favor <long|short>."

It must:
  - apply to FUTURE windows (do not name specific tickers, dates, or trades),
  - name a specific pattern or signal that recurred across the cluster,
  - fit in a single sentence, at most ~30 words.

Respond with ONLY the lesson text. No preamble, no quotes, no bullet points.
"""


def _format_case_block(case: TraceCase, idx: int) -> str:
    """Compact rendering of one cluster member for the cluster prompt."""
    votes = ", ".join(
        f"{a['name']}: {a['parsed_vote']:+.0f}" for a in case.analyst_outputs
    )
    return (
        f"--- Example {idx} ---\n"
        f"OHLCV window (t-0 most recent):\n{case.formatted_series}\n"
        f"Analyst votes: {votes}\n"
        f"Final action taken: {case.direction_taken}\n"
        f"Realised next-day return: {case.target_return:+.4%}  "
        f"(correct direction was {case.correct_direction})"
    )


def build_cluster_distill_prompt(cluster: List[TraceCase], kind: str) -> str:
    reps = cluster[:CLUSTER_REPS_PER_CLUSTER]
    blocks = [_format_case_block(c, i + 1) for i, c in enumerate(reps)]
    header = (
        f"This cluster contains {len(cluster)} similar "
        f"{'losing' if kind == 'bad' else 'winning'} trades. "
        f"Here are {len(reps)} representative examples (closest to the "
        f"cluster centroid):\n\n"
    )
    footer = (
        "\n\nBased on the COMMON pattern across these examples — not the "
        "details of any single one — distill ONE generalizable lesson now."
    )
    return header + "\n\n".join(blocks) + footer


def distill_lesson_from_cluster(judge: JudgeLLM,
                                cluster: List[TraceCase],
                                kind: str) -> str:
    """Distill one lesson from a cluster of similar cases.

    kind: 'bad' (loss cluster, allows abstain/confirmation lessons) or
          'good' (winning cluster, distil what worked)."""
    system = DISTILL_BAD_SYSTEM if kind == "bad" else DISTILL_GOOD_SYSTEM
    user = build_cluster_distill_prompt(cluster, kind)
    raw = judge.complete(system, user)
    lesson = raw.strip().strip('"').strip("'")
    lesson = re.sub(r"^[-*\d.\)\s]+", "", lesson).strip()
    return lesson


# ---------------------------------------------------------------------------
# Step 4 — integrate a new lesson into the bounded book
# ---------------------------------------------------------------------------

INTEGRATE_SYSTEM = """You are the curator of a small, bounded book of trading
lessons. The book changes RARELY: most new candidates are already covered
by what is in the book, or describe a different trading scenario that would
need its own slot. SKIP is the default; the bar for every other action is
high.

You will be given the current lessons (numbered) and ONE newly distilled
lesson. Choose EXACTLY ONE action:

  SKIP        - DEFAULT CHOICE. Use whenever the new lesson is already
                covered — even approximately — by an existing lesson, OR
                when you are not confident any other action clearly fits.

  UPDATE      - Use ONLY when the new lesson is about the SAME pattern,
                signal, or scenario as one existing lesson AND it adds one
                specific, missing detail (a threshold, an exception, an
                extra confirming signal). The updated text MUST be the
                existing lesson with a small additive edit — keep its
                wording largely intact. If you would need to rewrite the
                existing lesson substantially, the new candidate is a
                DIFFERENT rule: use SKIP (book is full) or APPEND (room).
                "Polishing the phrasing" alone is NOT a valid reason to
                UPDATE.

  APPEND      - Use only when the new lesson addresses a trading scenario
                or signal NOT covered by any existing lesson. Different
                wording about the same scenario is not enough — the
                underlying pattern must be different.

  CONTRADICT  - Use only for direct logical conflicts: under the same
                conditions, the two lessons prescribe opposite actions.
                Mere tension or different emphasis is not a contradiction.

Return ONLY a JSON object with this exact shape:

{
  "action": "SKIP" | "UPDATE" | "APPEND" | "CONTRADICT",
  "target_index": <int 0-based index of the existing lesson involved, or null>,
  "updated_lesson": <string with the merged lesson if action is UPDATE, else null>,
  "reasoning": <one short sentence explaining the choice>
}

Constraints:
- For SKIP and APPEND, target_index must be null.
- For UPDATE and CONTRADICT, target_index must be a valid index into the
  existing lessons list.
- For UPDATE, updated_lesson must be a single sentence <= ~30 words AND
  must share most of its wording with the existing lesson at
  target_index — it is an edit, not a rewrite. A reader should be able
  to look at the before and after and immediately see they describe the
  same rule with one added detail.
- Strong bias toward SKIP. If the new lesson restates a general principle
  already encoded in the book, SKIP.
"""


def build_integrate_user_prompt(existing: List[str], new_lesson: str) -> str:
    if existing:
        listed = "\n".join(f"  lesson_{i}: {l}" for i, l in enumerate(existing))
    else:
        listed = "  (none yet)"
    return (
        f"Existing lessons:\n{listed}\n\n"
        f"Newly distilled lesson:\n  {new_lesson}\n\n"
        f"Return your JSON decision now."
    )


def _extract_json(text: str) -> Dict[str, Any]:
    """Strip code fences and locate the JSON object in the judge's reply."""
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(cleaned[start:end + 1])


# Small built-in stopword set — keeps the similarity check focused on the
# content words of a lesson (the signals and actions, not the connectives).
_STOPWORDS = set("""
a an the and or but if then so to of in on at by for with from into onto upon
is are was were be been being am will would should could can may might must
do does did has have had not no nor as it its this that these those we our
you your they their he she his her i me my mine yours ours
when while where what which who whom how why because since though although
above below over under between among during before after near far against
about across against around between beyond through within without
than per up down off out also too very more most less least just only
each every any all some such own same other another both either neither
""".split())


def _content_words(s: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOPWORDS}


def _jaccard(a: str, b: str) -> float:
    """Bag-of-content-words Jaccard similarity. 0 = disjoint, 1 = identical."""
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def integrate_lesson(judge: JudgeLLM, book: LessonBook,
                     new_lesson: str,
                     cluster: List[TraceCase],
                     kind: str) -> Dict[str, Any]:
    """Run the judge's integration step and mutate `book` accordingly.

    Returns the audit-log entry that was appended to book.history.
    """
    raw = judge.complete(
        INTEGRATE_SYSTEM,
        build_integrate_user_prompt(book.lessons, new_lesson),
    )
    try:
        decision = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as e:
        decision = {"action": "SKIP", "target_index": None,
                    "updated_lesson": None,
                    "reasoning": f"Parse error: {e}; defaulting to SKIP."}

    action = (decision.get("action") or "SKIP").upper()
    idx = decision.get("target_index")
    updated = decision.get("updated_lesson")

    applied = action
    before = list(book.lessons)
    similarity: Optional[float] = None

    if action == "SKIP":
        pass

    elif action == "APPEND":
        if len(book.lessons) < book.num_lessons:
            book.lessons.append(new_lesson)
        else:
            applied = "APPEND_REJECTED_FULL"

    elif action == "UPDATE":
        if isinstance(idx, int) and 0 <= idx < len(book.lessons) and updated:
            original = book.lessons[idx]
            similarity = _jaccard(original, updated)
            if similarity < UPDATE_MIN_SIMILARITY:
                # Stealth rewrite — keep the book, skip the change.
                applied = (f"UPDATE_DOWNGRADED_TO_SKIP"
                           f"(sim={similarity:.2f}<{UPDATE_MIN_SIMILARITY})")
            else:
                book.lessons[idx] = updated.strip()
        else:
            applied = "UPDATE_REJECTED_BAD_INDEX"

    elif action == "CONTRADICT":
        if isinstance(idx, int) and 0 <= idx < len(book.lessons):
            book.lessons.pop(idx)   # drop existing
            # new lesson is also discarded by definition
        else:
            applied = "CONTRADICT_REJECTED_BAD_INDEX"

    else:
        applied = f"UNKNOWN_ACTION:{action}"

    entry = {
        "kind": kind,
        "cluster_size": len(cluster),
        "cluster_data_point_ids": [c.data_point_id for c in cluster],
        "distilled_lesson": new_lesson,
        "judge_decision": decision,
        "applied": applied,
        "update_similarity": similarity,
        "book_before": before,
        "book_after": list(book.lessons),
    }
    book.history.append(entry)
    return entry


# ---------------------------------------------------------------------------
# Step 5 — top-level driver
# ---------------------------------------------------------------------------

def _process_clusters(judge: JudgeLLM,
                      book: LessonBook,
                      clusters: List[List[TraceCase]],
                      kind: str) -> None:
    """Distill + integrate a list of clusters of the same kind."""
    label = "LOSS-CLUSTER" if kind == "bad" else "WIN-CLUSTER"
    for i, cluster in enumerate(clusters, 1):
        try:
            lesson = distill_lesson_from_cluster(judge, cluster, kind)
            if not lesson:
                print(f"  [{label} {i}/{len(clusters)}] empty lesson — skipping.")
                continue
            entry = integrate_lesson(judge, book, lesson, cluster, kind)
            print(f"  [{label} {i}/{len(clusters)}] size={len(cluster):>3}  "
                  f"{entry['applied']:<32}  |book|={len(book.lessons)}")
        except Exception as e:
            ids = [c.data_point_id for c in cluster[:3]]
            print(f"  [{label} {i}/{len(clusters)}] FAILED on cluster "
                  f"(first ids={ids}): {e}", file=sys.stderr)


def run_pipeline(traces_dir: str,
                 out_path: str,
                 num_lessons: int = NUM_LESSONS_DEFAULT,
                 min_cluster_size: int = CLUSTER_MIN_SIZE_DEFAULT,
                 target_cluster_size: int = CLUSTER_TARGET_SIZE,
                 include_successes: bool = True,
                 dry_run: bool = False,
                 limit: Optional[int] = None) -> LessonBook:
    print(f"[1/5] Loading traces from {traces_dir!r} ...")
    sets = load_cases(traces_dir)
    bad_cases, good_cases = sets["bad"], sets["good"]
    print(f"      bad (wrong-sign):  {len(bad_cases)} cases")
    print(f"      good (right-sign): {len(good_cases)} cases")

    if limit is not None:
        bad_cases = bad_cases[:limit]
        good_cases = good_cases[:limit]
        print(f"      limited to first {limit} of each kind for this run.")

    if not include_successes:
        good_cases = []
        print(f"      --include-successes disabled; ignoring good cases.")

    embedder = EmbeddingClient(dry_run=dry_run)

    print(f"[2/5] Clustering "
          f"(min_cluster_size={min_cluster_size}, "
          f"target_cluster_size={target_cluster_size}) ...")
    bad_clusters = cluster_cases(bad_cases, embedder,
                                 min_cluster_size=min_cluster_size,
                                 target_cluster_size=target_cluster_size)
    good_clusters = cluster_cases(good_cases, embedder,
                                  min_cluster_size=min_cluster_size,
                                  target_cluster_size=target_cluster_size)
    bad_dropped = len(bad_cases) - sum(len(c) for c in bad_clusters)
    good_dropped = len(good_cases) - sum(len(c) for c in good_clusters)
    print(f"      bad : {len(bad_clusters)} clusters kept  "
          f"({bad_dropped} cases dropped as singletons/noise)")
    print(f"      good: {len(good_clusters)} clusters kept  "
          f"({good_dropped} cases dropped as singletons/noise)")

    judge = JudgeLLM(dry_run=dry_run)
    book = LessonBook(num_lessons=num_lessons)

    print(f"[3/5] Distilling + integrating loss-cluster lessons "
          f"(judge={JUDGE_MODEL}, num_lessons={num_lessons}, "
          f"dry_run={dry_run}) ...")
    _process_clusters(judge, book, bad_clusters, "bad")

    if good_clusters:
        print(f"[4/5] Distilling + integrating win-cluster lessons ...")
        _process_clusters(judge, book, good_clusters, "good")
    else:
        print(f"[4/5] No win clusters to process.")

    print(f"[5/5] Saving lesson book to {out_path!r} ...")
    book.save(out_path)

    print(f"\nFinal lessons:")
    if not book.lessons:
        print("      (none)")
    for i, l in enumerate(book.lessons):
        print(f"  {i}. {l}")

    return book


def get_augmented_agent_prompt(base_prompt: str, book: LessonBook) -> str:
    """Append the current lessons to an analyst's system prompt.

    Plug the output of this directly into SHARED_AGENT_SYSTEM in the next
    training/inference run."""
    if not book.lessons:
        return base_prompt
    block = "\n\nLessons learned from prior runs (apply when relevant):\n"
    block += "\n".join(f"- {l}" for l in book.lessons)
    return base_prompt.rstrip() + block


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--traces-dir", default=".",
                   help="Directory containing data_point_*.json (default: cwd).")
    p.add_argument("--out", default="lessons.json",
                   help="Where to save the final lesson book + history.")
    p.add_argument("--num-lessons", type=int, default=NUM_LESSONS_DEFAULT,
                   help=f"Lesson-book cap (default {NUM_LESSONS_DEFAULT}).")
    p.add_argument("--min-cluster-size", type=int,
                   default=CLUSTER_MIN_SIZE_DEFAULT,
                   help=f"Drop clusters smaller than this "
                        f"(default {CLUSTER_MIN_SIZE_DEFAULT}).")
    p.add_argument("--target-cluster-size", type=int,
                   default=CLUSTER_TARGET_SIZE,
                   help=f"Used to auto-pick KMeans k = N/target "
                        f"(default {CLUSTER_TARGET_SIZE}).")
    p.add_argument("--include-successes", action="store_true", default=True,
                   help="Also distill lessons from winning trades (default: on).")
    p.add_argument("--no-successes", dest="include_successes",
                   action="store_false",
                   help="Disable success distillation; only learn from losses.")
    p.add_argument("--limit", type=int, default=None,
                   help="Only use the first N cases of each kind (debugging).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip real LLM/embedding calls; use deterministic stubs.")
    args = p.parse_args()

    run_pipeline(
        traces_dir=args.traces_dir,
        out_path=args.out,
        num_lessons=args.num_lessons,
        min_cluster_size=args.min_cluster_size,
        target_cluster_size=args.target_cluster_size,
        include_successes=args.include_successes,
        dry_run=args.dry_run,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()