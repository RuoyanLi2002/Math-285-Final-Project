from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol


JUDGE_MODEL = "claude-opus-4-8"
JUDGE_TEMPERATURE: Optional[float] = None
JUDGE_MAX_TOKENS = 1024
NUM_LESSONS_DEFAULT = 5
UPDATE_MIN_SIMILARITY = 0.5
EMBEDDING_MODEL = "voyage-finance-2"
CLUSTER_MIN_SIZE_DEFAULT = 3
CLUSTER_TARGET_SIZE = 5
CLUSTER_REPS_PER_CLUSTER = 3
RANDOM_SEED = 42


class CaseLike(Protocol):
    kind: str
    data_point_id: int
    def embed_text(self) -> str: ...


@dataclass
class LessonBook:
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


class JudgeLLM:
    def __init__(self, model: str = JUDGE_MODEL,
                 temperature: Optional[float] = JUDGE_TEMPERATURE,
                 max_tokens: int = JUDGE_MAX_TOKENS,
                 dry_run: bool = False):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dry_run = dry_run
        self.client = None
        if not dry_run:
            try:
                from anthropic import Anthropic
            except ImportError as e:
                raise ImportError("pip install anthropic") from e
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set.")
            self.client = Anthropic(api_key=api_key)

    def complete(self, system: str, user: str) -> str:
        if self.dry_run:
            return self._fake_response(system, user)
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [
                {"role": "user", "content": user},
            ],
        }
        
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        resp = self.client.messages.create(**kwargs)
        
        return "".join(
            block.text for block in resp.content
            if getattr(block, "type", None) == "text"
        )

    @staticmethod
    def _fake_response(system: str, user: str) -> str:
        if "Return ONLY a JSON object" in system:
            decision = "APPEND" if "lesson_1" not in user else "SKIP"
            return json.dumps({
                "action": decision, "target_index": None,
                "updated_lesson": None, "reasoning": "[dry-run stub]",
            })
        return ("When recent bars show a strong directional move with rising "
                "volume, do not fade it without an explicit reversal signal.")


class EmbeddingClient:
    MAX_BATCH_TOKENS = 100_000
    MAX_INPUTS = 128
    MAX_TOKENS_PER_TEXT = 31_000

    def __init__(self, model: str = EMBEDDING_MODEL, dry_run: bool = False,
                 batch_size: int = 64):
        self.model = model
        self.dry_run = dry_run
        self.batch_size = batch_size
        self.client = None
        if not dry_run:
            try:
                import voyageai
            except ImportError as e:
                raise ImportError("pip install voyageai") from e
            api_key = os.environ.get("VOYAGE_API_KEY")
            if not api_key:
                raise ValueError("VOYAGE_API_KEY is not set.")
            self.client = voyageai.Client(api_key=api_key)

    def _count_tokens(self, text: str) -> int:
        if self.client is not None:
            try:
                return self.client.count_tokens([text], model=self.model)
            except Exception:
                pass
        return max(1, len(text) // 4 + 1)

    def _truncate(self, text: str) -> str:
        if self._count_tokens(text) <= self.MAX_TOKENS_PER_TEXT:
            return text
        
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count_tokens(text[:mid]) <= self.MAX_TOKENS_PER_TEXT:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo]

    def _batches(self, texts: List[str]):
        batch: List[str] = []
        batch_tokens = 0
        for t in texts:
            tok = self._count_tokens(t)
            would_overflow = (
                batch and (
                    batch_tokens + tok > self.MAX_BATCH_TOKENS
                    or len(batch) >= self.MAX_INPUTS
                )
            )
            if would_overflow:
                yield batch
                batch, batch_tokens = [], 0
            batch.append(t)
            batch_tokens += tok
        if batch:
            yield batch

    def embed(self, texts: List[str]):
        import numpy as np
        if not texts:
            return np.zeros((0, 8), dtype=np.float32)
        if self.dry_run:
            vecs = []
            for t in texts:
                local = np.random.default_rng(abs(hash(t)) % (2**32))
                vecs.append(local.standard_normal(64).astype(np.float32))
            arr = np.stack(vecs, axis=0)
        else:
            texts = [self._truncate(t) for t in texts]
            rows = []
            for batch in self._batches(texts):
                resp = self.client.embed(
                    batch, model=self.model, input_type="document"
                )
                rows.extend(resp.embeddings)
            arr = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms



def cluster_cases(cases: List[CaseLike],
                  embedder: EmbeddingClient,
                  min_cluster_size: int = CLUSTER_MIN_SIZE_DEFAULT,
                  target_cluster_size: int = CLUSTER_TARGET_SIZE,
                  ) -> List[List[CaseLike]]:
    if not cases:
        return []
    if len(cases) < 2 * min_cluster_size:
        return [cases] if len(cases) >= min_cluster_size else []

    try:
        from sklearn.cluster import KMeans
    except ImportError as e:
        raise ImportError(
            "Clustering requires scikit-learn. Install: pip install scikit-learn"
        ) from e

    import numpy as np
    embeddings = embedder.embed([c.embed_text() for c in cases])
    k = max(2, len(cases) // target_cluster_size)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(embeddings)
    labels = km.labels_

    buckets: Dict[int, List[int]] = {}
    for idx, lbl in enumerate(labels):
        buckets.setdefault(int(lbl), []).append(idx)

    clusters: List[List[CaseLike]] = []
    for lbl, indices in buckets.items():
        if len(indices) < min_cluster_size:
            continue
        centroid = embeddings[indices].mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) or 1.0)
        sims = embeddings[indices] @ centroid
        order = np.argsort(-sims)
        clusters.append([cases[indices[i]] for i in order])

    clusters.sort(key=len, reverse=True)
    return clusters



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
- For UPDATE and CONTRADICT, target_index must be a valid index.
- For UPDATE, updated_lesson must be a single sentence <= ~30 words AND
  must share most of its wording with the existing lesson at target_index.
- Strong bias toward SKIP.
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
    cleaned = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    return json.loads(cleaned[start:end + 1])


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
    wa, wb = _content_words(a), _content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def integrate_lesson(judge: JudgeLLM, book: LessonBook,
                     new_lesson: str, cluster: List[CaseLike],
                     kind: str) -> Dict[str, Any]:
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
                applied = (f"UPDATE_DOWNGRADED_TO_SKIP"
                           f"(sim={similarity:.2f}<{UPDATE_MIN_SIMILARITY})")
            else:
                book.lessons[idx] = updated.strip()
        else:
            applied = "UPDATE_REJECTED_BAD_INDEX"
    elif action == "CONTRADICT":
        if isinstance(idx, int) and 0 <= idx < len(book.lessons):
            book.lessons.pop(idx)
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



LoadCasesFn = Callable[[str], Dict[str, List[CaseLike]]]
DistillFn   = Callable[[JudgeLLM, List[CaseLike], str], str]


def _process_clusters(judge: JudgeLLM, book: LessonBook,
                      clusters: List[List[CaseLike]], kind: str,
                      distill_fn: DistillFn) -> None:
    label = "LOSS-CLUSTER" if kind == "bad" else "WIN-CLUSTER"
    for i, cluster in enumerate(clusters, 1):
        try:
            lesson = distill_fn(judge, cluster, kind)
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


def run_pipeline(load_cases_fn: LoadCasesFn,
                 distill_fn: DistillFn,
                 traces_dir: str,
                 out_path: str,
                 model_label: str = "model",
                 num_lessons: int = NUM_LESSONS_DEFAULT,
                 min_cluster_size: int = CLUSTER_MIN_SIZE_DEFAULT,
                 target_cluster_size: int = CLUSTER_TARGET_SIZE,
                 include_successes: bool = True,
                 dry_run: bool = False,
                 limit: Optional[int] = None) -> LessonBook:
    print(f"[1/5] Loading {model_label} traces from {traces_dir!r} ...")
    sets = load_cases_fn(traces_dir)
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

    print(f"[2/5] Clustering (min_size={min_cluster_size}, "
          f"target_size={target_cluster_size}) ...")
    bad_clusters = cluster_cases(bad_cases, embedder,
                                 min_cluster_size, target_cluster_size)
    good_clusters = cluster_cases(good_cases, embedder,
                                  min_cluster_size, target_cluster_size)
    print(f"      bad : {len(bad_clusters)} clusters kept "
          f"({len(bad_cases) - sum(len(c) for c in bad_clusters)} dropped)")
    print(f"      good: {len(good_clusters)} clusters kept "
          f"({len(good_cases) - sum(len(c) for c in good_clusters)} dropped)")

    judge = JudgeLLM(dry_run=dry_run)
    book = LessonBook(num_lessons=num_lessons)

    print(f"[3/5] Distilling + integrating loss-cluster lessons "
          f"(judge={JUDGE_MODEL}, num_lessons={num_lessons}, "
          f"dry_run={dry_run}) ...")
    _process_clusters(judge, book, bad_clusters, "bad", distill_fn)

    if good_clusters:
        print(f"[4/5] Distilling + integrating win-cluster lessons ...")
        _process_clusters(judge, book, good_clusters, "good", distill_fn)
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
    if not book.lessons:
        return base_prompt
    block = "\n\nLessons learned from prior runs (apply when relevant):\n"
    block += "\n".join(f"- {l}" for l in book.lessons)
    return base_prompt.rstrip() + block



def build_argparser(model_label: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=f"Distill lessons from {model_label} traces.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--traces-dir", default=".",
                   help="Directory containing data_point_*.json (default: cwd).")
    p.add_argument("--out", default=f"lessons_{model_label.lower()}.json",
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
                   help="Also distill lessons from winning trades (default on).")
    p.add_argument("--no-successes", dest="include_successes",
                   action="store_false")
    p.add_argument("--limit", type=int, default=None,
                   help="Only use the first N cases of each kind (debugging).")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip real LLM/embedding calls; use deterministic stubs.")
    return p