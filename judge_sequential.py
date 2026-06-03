from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List

from judge_core import (
    JudgeLLM, LessonBook, run_pipeline, build_argparser,
    CLUSTER_REPS_PER_CLUSTER,
)



@dataclass
class SequentialCase:
    kind: str
    path: str
    data_point_id: int
    final_decision: float
    target_return: float
    formatted_series: str

    generator_response: str
    generator_vote: float

    critic_response: str
    critic_strengths: str
    critic_serious_issues: str
    critic_has_serious_issues: bool

    refiner_response: str

    @property
    def direction_taken(self) -> str:
        if self.final_decision > 0: return "LONG (+1)"
        if self.final_decision < 0: return "SHORT (-1)"
        return "NO ACTION (0)"

    @property
    def correct_direction(self) -> str:
        return "LONG (+1)" if self.target_return > 0 else "SHORT (-1)"

    @property
    def generator_was_correct(self) -> bool:
        if self.generator_vote == 0:
            return False
        return (self.generator_vote > 0) == (self.target_return > 0)

    @property
    def refiner_changed_vote(self) -> bool:
        return self.generator_vote != self.final_decision

    def classify_failure_mode(self) -> str:
        if self.kind != "bad":
            return "N/A"
        gen_ok = self.generator_was_correct
        has_issues = self.critic_has_serious_issues
        changed = self.refiner_changed_vote

        if not gen_ok and not has_issues and not changed:
            return "CRITIC-FALSE-NEGATIVE"
        
        if gen_ok and has_issues and changed:
            return "CRITIC-FALSE-POSITIVE"
        
        if not gen_ok and has_issues and not changed:
            return "REFINER-FAIL"
        
        if has_issues and changed:
            return "REFINER-FAIL"
        
        if not gen_ok and not has_issues and changed:
            return "REFINER-FAIL"
        
        return "GENERATOR-FAIL"

    @property
    def pipeline_summary(self) -> str:
        gen = f"{self.generator_vote:+.0f}"
        crit = "ISSUES" if self.critic_has_serious_issues else "NO_ISSUES"
        ref = (f"{self.final_decision:+.0f}" if self.final_decision != 0
               else "0")
        changed = " [CHANGED]" if self.refiner_changed_vote else " [upheld]"
        return f"gen={gen} -> critic={crit} -> refiner={ref}{changed}"

    def embed_text(self) -> str:
        mode = self.classify_failure_mode()
        issues = (self.critic_serious_issues.strip()[:300]
                  if self.critic_has_serious_issues else "NO_ISSUES")
        return (
            f"{self.formatted_series}\n\n"
            f"Stage failure (heuristic): {mode}\n"
            f"Pipeline: {self.pipeline_summary}\n\n"
            f"Generator reasoning:\n{self.generator_response.strip()[:500]}\n\n"
            f"Critic serious issues: {issues}\n\n"
            f"Refiner reasoning:\n{self.refiner_response.strip()[:500]}"
        )



def load_cases(traces_dir: str) -> Dict[str, List[SequentialCase]]:
    pattern = os.path.join(traces_dir, "data_point_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matched {pattern!r}.")

    bad: List[SequentialCase] = []
    good: List[SequentialCase] = []
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

        try:
            gen = trace["generator"]
            critic = trace["critic"]
            refiner = trace["refiner"]
            cp = critic.get("parsed", {}) or {}
            case = SequentialCase(
                kind=kind, path=p,
                data_point_id=int(trace.get("data_point_id", -1)),
                final_decision=final, target_return=target,
                formatted_series=trace["input"]["formatted_series"],

                generator_response=gen.get("raw_response", ""),
                generator_vote=float(gen.get("parsed_vote", 0.0)),

                critic_response=critic.get("raw_response", ""),
                critic_strengths=cp.get("strengths", ""),
                critic_serious_issues=cp.get("serious_issues", ""),
                critic_has_serious_issues=bool(cp.get("has_serious_issues", False)),

                refiner_response=refiner.get("raw_response", ""),
            )
        except (KeyError, TypeError) as e:
            print(f"[skip] {p}: malformed Sequential trace ({e})", file=sys.stderr)
            continue

        (good if same_sign else bad).append(case)

    return {"bad": bad, "good": good}



DISTILL_BAD_SYSTEM = """You are a trading post-mortem judge analyzing a
SEQUENTIAL pipeline:

  GENERATOR : produces an initial directional call from the data alone.
  CRITIC    : audits the call; produces "strengths" and "serious_issues"
              (no vote).
  REFINER   : produces the final decision given data + generator + critic.

You will be shown a CLUSTER of similar losing trades. Each trade had the
wrong directional sign vs the realised return.

First, identify which STAGE is most responsible across the cluster
(a heuristic label is provided per case to help):

  GENERATOR-FAIL        : initial call wrong; critic missed it; refiner
                          upheld. The analyst-level signal interpretation
                          is the problem.
  CRITIC-FALSE-POSITIVE : generator was right; critic invented serious
                          issues; refiner flipped to the wrong side.
  CRITIC-FALSE-NEGATIVE : generator was wrong AND real issues were
                          visible in the data, but critic did not flag.
  REFINER-FAIL          : critic correctly raised issues but refiner did
                          not adjust, or critic raised none but refiner
                          overrode anyway to the wrong side.

Then produce ONE short, generalizable LESSON whose SHAPE matches the
failing stage:

  GENERATOR-FAIL:
    DIRECTIONAL  : "When <signal>, favor <long|short>."
    ABSTAIN      : "When <signal>, output 0 (no action)."
    CONFIRMATION : "Do not take a <long|short> on <signal> unless
                    <additional confirmation> is also present."

  CRITIC-FALSE-POSITIVE (critic too aggressive):
    "Critic: do not flag <pattern X> as a serious issue when <Y>."

  CRITIC-FALSE-NEGATIVE (critic too lenient):
    "Critic: flag <pattern X> as a serious issue when <Y is present>."

  REFINER-FAIL:
    "Refiner: when the critic <action>, <recommended response>."

Return a JSON object with this exact shape:

{
  "target_stage": "generator" | "critic" | "refiner",
  "failure_mode": "GENERATOR-FAIL" | "CRITIC-FALSE-POSITIVE"
                | "CRITIC-FALSE-NEGATIVE" | "REFINER-FAIL",
  "lesson": "<the lesson, one sentence, <= ~30 words>"
}

The lesson must apply to FUTURE windows (no tickers/dates), name a
specific pattern, fit in one sentence <= ~30 words.
Return ONLY the JSON object. No preamble, no code fences.
"""


DISTILL_GOOD_SYSTEM = """You are a trading post-mortem judge analyzing a
SEQUENTIAL pipeline (Generator -> Critic -> Refiner). You will be shown a
CLUSTER of similar WINNING trades.

Identify whether the wins came from:
  - generator getting it right with critic finding no issues (and refiner
    upholding) — the easiest case;
  - critic correctly flagging a marginal initial call, refiner adjusting
    — the pipeline doing its job;
  - critic finding no issues despite a noisy setup — restraint that paid off.

Return a JSON object:

{
  "target_stage": "generator" | "critic" | "refiner",
  "failure_mode": "N/A",
  "lesson": "<one positive directional rule, <= ~30 words>"
}

The lesson should be phrased as a positive rule for the indicated stage,
e.g. for generator: "When <signal>, favor <long|short>."
Return ONLY the JSON object.
"""



def _format_case_block(case: SequentialCase, idx: int) -> str:
    mode = case.classify_failure_mode() if case.kind == "bad" else "(win)"
    issues = (case.critic_serious_issues.strip()[:400]
              if case.critic_has_serious_issues else "NO SERIOUS ISSUES")
    return (
        f"--- Example {idx}  [stage failure hint: {mode}] ---\n"
        f"OHLCV+indicator window (t-0 most recent):\n{case.formatted_series}\n\n"
        f"Pipeline: {case.pipeline_summary}\n\n"
        f"Generator reasoning (excerpt):\n{case.generator_response.strip()[:450]}\n\n"
        f"Critic serious_issues: {issues}\n\n"
        f"Refiner reasoning (excerpt):\n{case.refiner_response.strip()[:450]}\n\n"
        f"Final decision: {case.direction_taken}\n"
        f"Realised return: {case.target_return:+.4%}  "
        f"(correct direction was {case.correct_direction})"
    )


def build_cluster_distill_prompt(cluster: List[SequentialCase], kind: str) -> str:
    reps = cluster[:CLUSTER_REPS_PER_CLUSTER]
    blocks = [_format_case_block(c, i + 1) for i, c in enumerate(reps)]

    if kind == "bad":
        from collections import Counter
        mode_counts = Counter(c.classify_failure_mode() for c in cluster)
        modes_summary = ", ".join(f"{m}={n}" for m, n in mode_counts.most_common())
        header = (
            f"This cluster contains {len(cluster)} similar losing pipeline "
            f"runs. Heuristic stage-failure distribution: {modes_summary}.\n\n"
            f"Here are {len(reps)} representative examples "
            f"(closest to centroid):\n\n"
        )
        footer = (
            "\n\nFirst confirm or override the dominant failing stage, then "
            "produce the JSON object with target_stage / failure_mode / lesson."
        )
    else:
        header = (
            f"This cluster contains {len(cluster)} similar winning pipeline "
            f"runs. Here are {len(reps)} representative examples:\n\n"
        )
        footer = (
            "\n\nProduce the JSON object capturing the pattern that worked."
        )

    return header + "\n\n".join(blocks) + footer


def _parse_distilled_lesson(raw: str) -> Dict[str, str]:
    cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1:
        return {"target_stage": "generator", "failure_mode": "PARSE_ERROR",
                "lesson": cleaned[:200]}
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return {"target_stage": "generator", "failure_mode": "PARSE_ERROR",
                "lesson": cleaned[:200]}
    return {
        "target_stage": str(parsed.get("target_stage", "generator")).lower(),
        "failure_mode": str(parsed.get("failure_mode", "")),
        "lesson": str(parsed.get("lesson", "")).strip(),
    }


def distill_lesson(judge: JudgeLLM, cluster: List[SequentialCase],
                   kind: str) -> str:
    system = DISTILL_BAD_SYSTEM if kind == "bad" else DISTILL_GOOD_SYSTEM
    user = build_cluster_distill_prompt(cluster, kind)
    raw = judge.complete(system, user)
    payload = _parse_distilled_lesson(raw)

    cluster[0]._last_lesson_meta = payload

    lesson = payload["lesson"]
    if payload["target_stage"] != "generator":
        lesson = f"[{payload['target_stage'].upper()}] {lesson}"
    return lesson



def split_lessons_by_stage(book: LessonBook) -> Dict[str, List[str]]:
    out = {"generator": [], "critic": [], "refiner": []}
    for l in book.lessons:
        m = re.match(r"^\[(GENERATOR|CRITIC|REFINER)\]\s*(.*)$", l)
        if m:
            stage = m.group(1).lower()
            out[stage].append(m.group(2).strip())
        else:
            out["generator"].append(l)
    return out


def get_augmented_stage_prompts(base_prompts: Dict[str, str],
                                book: LessonBook) -> Dict[str, str]:
    by_stage = split_lessons_by_stage(book)
    out: Dict[str, str] = {}
    for stage, base in base_prompts.items():
        lessons = by_stage.get(stage, [])
        if not lessons:
            out[stage] = base
            continue
        block = (f"\n\nLessons learned from prior runs "
                 f"(apply when relevant):\n")
        block += "\n".join(f"- {l}" for l in lessons)
        out[stage] = base.rstrip() + block
    return out



def main():
    args = build_argparser("Sequential").parse_args()
    run_pipeline(
        load_cases_fn=load_cases,
        distill_fn=distill_lesson,
        model_label="Sequential",
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
