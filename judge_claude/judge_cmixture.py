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
class CMixtureCase:
    kind: str
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

    @property
    def analyst_votes(self) -> List[float]:
        return [float(a["parsed_vote"]) for a in self.analyst_outputs]

    @property
    def is_consensus(self) -> bool:
        """All three analysts voted identically."""
        v = self.analyst_votes
        return len(v) > 0 and len(set(v)) == 1

    @property
    def consensus_label(self) -> str:
        return "UNANIMOUS" if self.is_consensus else "SPLIT"

    def embed_text(self) -> str:
        votes_str = ",".join(f"{v:+.0f}" for v in self.analyst_votes)
        return (
            f"{self.formatted_series}\n\n"
            f"Analyst votes: [{votes_str}]  ({self.consensus_label})\n\n"
            f"Orchestrator reasoning:\n{self.orchestrator_response.strip()}"
        )



def load_cases(traces_dir: str) -> Dict[str, List[CMixtureCase]]:
    pattern = os.path.join(traces_dir, "data_point_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matched {pattern!r}.")

    bad: List[CMixtureCase] = []
    good: List[CMixtureCase] = []
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
            case = CMixtureCase(
                kind=kind, path=p,
                data_point_id=int(trace.get("data_point_id", -1)),
                final_decision=final, target_return=target,
                formatted_series=trace["input"]["formatted_series"],
                analyst_outputs=trace["analysts"]["outputs"],
                orchestrator_response=trace["orchestrator"]["raw_response"],
            )
        except (KeyError, TypeError) as e:
            print(f"[skip] {p}: malformed CMixture trace ({e})", file=sys.stderr)
            continue

        (good if same_sign else bad).append(case)

    return {"bad": bad, "good": good}



DISTILL_BAD_SYSTEM = """You are a trading post-mortem judge analyzing a
multi-agent ENSEMBLE system: three independent analysts each vote
(LONG/SHORT/NO ACTION) on the same OHLCV+indicator window, and an
orchestrator combines the votes into a final decision.

You will be shown a CLUSTER of similar losing trades. Each trade had the
wrong directional sign vs the realised return.

First, identify the dominant failure mode across the cluster:

  CONSENSUS-ERROR  : all three analysts voted the same wrong direction;
                     the orchestrator went along. The failure is in the
                     analyst-level signal interpretation.
  SPLIT-RESOLUTION : analysts split (mixed votes), and the orchestrator
                     picked the wrong side. The failure is in conflict
                     resolution under ambiguity.

Then produce ONE short, generalizable LESSON that, if the analysts (and
orchestrator) had followed it, would have made this kind of mistake less
likely on FUTURE windows.

Lesson shape rules:
  - For CONSENSUS-ERROR clusters: prefer a DIRECTIONAL ("favor X when Y"),
    CONFIRMATION ("do not take X unless Z"), or ABSTAIN lesson, depending
    on whether the data points the other way, suggests caution, or shows
    genuine ambiguity.
  - For SPLIT-RESOLUTION clusters: STRONGLY prefer an ABSTAIN lesson
    ("when analysts split on <pattern>, output 0 rather than picking a
    side"). Resolving split votes the wrong way is a sign the situation
    was genuinely ambiguous.

Lesson shapes:
  DIRECTIONAL : "When <signal>, favor <long|short>."
  ABSTAIN     : "When <signal>, output 0 (no action)."
  CONFIRMATION: "Do not take a <long|short> on <signal> unless
                 <additional confirmation> is also present."

The lesson must apply to FUTURE windows (no specific tickers, dates),
name a specific pattern or signal, fit in one sentence <= ~30 words.
Respond with ONLY the lesson text. No preamble, no quotes, no bullets.
"""


DISTILL_GOOD_SYSTEM = """You are a trading post-mortem judge analyzing a
multi-agent ENSEMBLE system: three independent analysts each vote
(LONG/SHORT/NO ACTION) on the same OHLCV+indicator window, and an
orchestrator combines the votes into a final decision.

You will be shown a CLUSTER of similar WINNING trades. Each trade took a
directional position whose sign matched the realised return.

Note whether the wins came from analyst CONSENSUS (all three agreed) or
from a SPLIT that the orchestrator resolved correctly. Consensus wins
are usually stronger evidence of a generalizable pattern.

Produce ONE short, generalizable LESSON that captures the pattern the
analysts correctly identified, phrased as a directional rule:

  "When <signal>, favor <long|short>."

The lesson must apply to FUTURE windows, name a specific pattern or
signal that recurred across the cluster, fit in one sentence <= ~30 words.
Respond with ONLY the lesson text.
"""



def _format_case_block(case: CMixtureCase, idx: int) -> str:
    votes = ", ".join(
        f"{a['name']}: {a['parsed_vote']:+.0f}" for a in case.analyst_outputs
    )
    return (
        f"--- Example {idx}  [{case.consensus_label} on {case.direction_taken}] ---\n"
        f"OHLCV+indicator window (t-0 most recent):\n{case.formatted_series}\n\n"
        f"Analyst votes: {votes}\n"
        f"Orchestrator final action: {case.direction_taken}\n"
        f"Realised return: {case.target_return:+.4%}  "
        f"(correct direction was {case.correct_direction})"
    )


def build_cluster_distill_prompt(cluster: List[CMixtureCase], kind: str) -> str:
    reps = cluster[:CLUSTER_REPS_PER_CLUSTER]
    blocks = [_format_case_block(c, i + 1) for i, c in enumerate(reps)]

    consensus_count = sum(1 for c in cluster if c.is_consensus)
    split_count = len(cluster) - consensus_count
    pattern_summary = (
        f"Across the full cluster of {len(cluster)} cases: "
        f"{consensus_count} unanimous, {split_count} split.\n\n"
    )

    header = (
        f"This cluster contains {len(cluster)} similar "
        f"{'losing' if kind == 'bad' else 'winning'} trades. "
        f"{pattern_summary}"
        f"Here are {len(reps)} representative examples (closest to centroid):\n\n"
    )
    footer = (
        "\n\nFirst identify the dominant failure mode (CONSENSUS-ERROR or "
        "SPLIT-RESOLUTION) using the unanimous/split counts above, then "
        "distill ONE generalizable lesson."
    ) if kind == "bad" else (
        "\n\nBased on the COMMON pattern across these examples, distill "
        "ONE generalizable lesson now."
    )
    return header + "\n\n".join(blocks) + footer


def distill_lesson(judge: JudgeLLM, cluster: List[CMixtureCase],
                   kind: str) -> str:
    system = DISTILL_BAD_SYSTEM if kind == "bad" else DISTILL_GOOD_SYSTEM
    user = build_cluster_distill_prompt(cluster, kind)
    raw = judge.complete(system, user)
    lesson = raw.strip().strip('"').strip("'")
    lesson = re.sub(r"^[-*\d.\)\s]+", "", lesson).strip()
    return lesson



def main():
    args = build_argparser("CMixture").parse_args()
    run_pipeline(
        load_cases_fn=load_cases,
        distill_fn=distill_lesson,
        model_label="CMixture",
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