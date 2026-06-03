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
class DebateCase:
    kind: str
    path: str
    data_point_id: int
    final_decision: float
    target_return: float
    formatted_series: str
    num_rounds: int
    rounds: List[List[Dict[str, Any]]]

    @property
    def direction_taken(self) -> str:
        return "LONG (+1)" if self.final_decision > 0 else "SHORT (-1)"

    @property
    def correct_direction(self) -> str:
        return "LONG (+1)" if self.target_return > 0 else "SHORT (-1)"

    @property
    def num_analysts(self) -> int:
        return len(self.rounds[0]) if self.rounds else 0

    def vote_at(self, round_idx: int, analyst_idx: int) -> float:
        return float(self.rounds[round_idx][analyst_idx]["parsed_vote"])

    @property
    def round_1_votes(self) -> List[float]:
        return [float(o["parsed_vote"]) for o in self.rounds[0]]

    @property
    def final_round_votes(self) -> List[float]:
        return [float(o["parsed_vote"]) for o in self.rounds[-1]]

    @property
    def per_analyst_progression(self) -> List[List[float]]:
        """[[r1, r2, ..., rN], ...] per analyst."""
        return [[self.vote_at(r, i) for r in range(self.num_rounds)]
                for i in range(self.num_analysts)]

    @property
    def num_flippers(self) -> int:
        return sum(1 for prog in self.per_analyst_progression
                   if prog[0] != prog[-1])

    @property
    def num_correct_in_round_1(self) -> int:
        target_sign = 1 if self.target_return > 0 else -1
        return sum(1 for v in self.round_1_votes if v == target_sign)

    def classify_failure_mode(self) -> str:
        if self.kind != "bad":
            return "N/A"
        r1_correct = self.num_correct_in_round_1
        flippers   = self.num_flippers
        r1_unanimous_wrong = (
            r1_correct == 0
            and len(set(self.round_1_votes)) == 1
            and self.round_1_votes[0] != 0
        )
        if r1_unanimous_wrong:
            return "ROUND-1 ERROR"
        if r1_correct >= 1 and flippers >= 1:
            target_sign = 1 if self.target_return > 0 else -1
            flipped_away = sum(
                1 for prog in self.per_analyst_progression
                if prog[0] == target_sign and prog[-1] != target_sign
            )
            if flipped_away >= 1:
                return "DRIFT"
        final_unanimous = len(set(self.final_round_votes)) == 1
        if final_unanimous and r1_correct >= 1:
            return "GROUPTHINK"
        return "HUNG"

    @property
    def progression_summary(self) -> str:
        """Render per-analyst round-1 -> final vote sequences."""
        lines = []
        for i in range(self.num_analysts):
            name = self.rounds[0][i]["name"]
            seq = " -> ".join(f"{self.vote_at(r, i):+.0f}"
                              for r in range(self.num_rounds))
            lines.append(f"  {name}: {seq}")
        return "\n".join(lines)

    def embed_text(self) -> str:
        final_reasoning = "\n".join(
            o["raw_response"].strip()[:300] for o in self.rounds[-1]
        )
        return (
            f"{self.formatted_series}\n\n"
            f"Failure mode (heuristic): {self.classify_failure_mode()}\n"
            f"Vote progression:\n{self.progression_summary}\n\n"
            f"Final-round reasoning:\n{final_reasoning}"
        )



def load_cases(traces_dir: str) -> Dict[str, List[DebateCase]]:
    pattern = os.path.join(traces_dir, "data_point_*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No trace files matched {pattern!r}.")

    bad: List[DebateCase] = []
    good: List[DebateCase] = []
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
            rounds_blk = trace["analysts"]["rounds"]
            rounds_list = [r["outputs"] for r in rounds_blk]
            case = DebateCase(
                kind=kind, path=p,
                data_point_id=int(trace.get("data_point_id", -1)),
                final_decision=final, target_return=target,
                formatted_series=trace["input"]["formatted_series"],
                num_rounds=len(rounds_list),
                rounds=rounds_list,
            )
        except (KeyError, TypeError) as e:
            print(f"[skip] {p}: malformed Debate trace ({e})", file=sys.stderr)
            continue

        (good if same_sign else bad).append(case)

    return {"bad": bad, "good": good}



DISTILL_BAD_SYSTEM = """You are a trading post-mortem judge analyzing a
multi-round DEBATE system: analysts cast votes in round 1, then from
round 2 onward see each other's previous-round outputs and may revise.
The final decision is the MAJORITY VOTE across the last round.

You will be shown a CLUSTER of similar losing trades, each with the
per-analyst vote progression (round-1 -> ... -> final).

First, identify which debate FAILURE MODE dominates across the cluster:

  GROUPTHINK    : at least one analyst was correct in round 1, but later
                  rounds converged to the wrong final consensus.
  DRIFT         : at least one analyst was correct in round 1 and FLIPPED
                  to the wrong side after seeing the others.
  HUNG          : analysts kept oscillating or stayed split; the final
                  majority happened to fall on the wrong side.
  ROUND-1 ERROR : all analysts agreed on the wrong direction in round 1;
                  no one disagreed so the debate could not correct it.

(A heuristic label per case is provided to help, but you may override.)

Then produce ONE short, generalizable LESSON. The lesson SHAPE should
match the failure mode:

  GROUPTHINK or ROUND-1 ERROR:
    DIRECTIONAL  : "When <signal>, favor <long|short>."
    CONFIRMATION : "Do not take a <long|short> on <signal> unless
                    <additional confirmation> is also present."
  DRIFT:
    STAND-GROUND : "When round-1 analysis shows <signal>, do not flip
                    vote in later rounds without <new evidence>."
  HUNG:
    ABSTAIN      : "When analysts oscillate or stay split on <signal>,
                    output 0 (no action)."

The lesson must apply to FUTURE windows (no specific tickers/dates),
name a specific pattern, fit in one sentence <= ~30 words.
Respond with ONLY the lesson text. No preamble, no quotes, no bullets.
"""


DISTILL_GOOD_SYSTEM = """You are a trading post-mortem judge analyzing a
multi-round DEBATE system. You will be shown a CLUSTER of similar WINNING
trades, each with the per-analyst vote progression.

Note whether the wins came from:
  - early consensus (analysts agreed in round 1 and stayed),
  - convergence (analysts split in round 1 but reached the correct view
    through debate), or
  - persistent disagreement that the majority resolved correctly.

Produce ONE short, generalizable LESSON capturing the pattern the
analysts correctly identified, phrased as a directional rule:

  "When <signal>, favor <long|short>."

The lesson must apply to FUTURE windows, name a specific pattern that
recurred across the cluster, fit in one sentence <= ~30 words.
Respond with ONLY the lesson text.
"""



def _format_case_block(case: DebateCase, idx: int) -> str:
    mode = case.classify_failure_mode() if case.kind == "bad" else "(win)"
    final_excerpts = "\n".join(
        f"  {o['name']}: {o['raw_response'].strip()[:350]}"
        for o in case.rounds[-1]
    )
    return (
        f"--- Example {idx}  [failure mode hint: {mode}] ---\n"
        f"OHLCV+indicator window (t-0 most recent):\n{case.formatted_series}\n\n"
        f"Per-analyst vote progression (r1 -> ... -> final):\n"
        f"{case.progression_summary}\n\n"
        f"Final-round reasoning excerpts:\n{final_excerpts}\n\n"
        f"Final majority decision: {case.direction_taken}\n"
        f"Realised return: {case.target_return:+.4%}  "
        f"(correct direction was {case.correct_direction})"
    )


def build_cluster_distill_prompt(cluster: List[DebateCase], kind: str) -> str:
    reps = cluster[:CLUSTER_REPS_PER_CLUSTER]
    blocks = [_format_case_block(c, i + 1) for i, c in enumerate(reps)]

    if kind == "bad":
        from collections import Counter
        mode_counts = Counter(c.classify_failure_mode() for c in cluster)
        modes_summary = ", ".join(f"{m}={n}" for m, n in mode_counts.most_common())
        header = (
            f"This cluster contains {len(cluster)} similar losing debates. "
            f"Heuristic failure-mode distribution: {modes_summary}.\n\n"
            f"Here are {len(reps)} representative examples "
            f"(closest to centroid):\n\n"
        )
        footer = (
            "\n\nFirst confirm or override the dominant failure mode "
            "(GROUPTHINK / DRIFT / HUNG / ROUND-1 ERROR), then distill ONE "
            "generalizable lesson whose shape matches that mode."
        )
    else:
        header = (
            f"This cluster contains {len(cluster)} similar winning debates. "
            f"Here are {len(reps)} representative examples:\n\n"
        )
        footer = (
            "\n\nBased on the COMMON pattern across these examples, distill "
            "ONE generalizable lesson now."
        )

    return header + "\n\n".join(blocks) + footer


def distill_lesson(judge: JudgeLLM, cluster: List[DebateCase],
                   kind: str) -> str:
    system = DISTILL_BAD_SYSTEM if kind == "bad" else DISTILL_GOOD_SYSTEM
    user = build_cluster_distill_prompt(cluster, kind)
    raw = judge.complete(system, user)
    lesson = raw.strip().strip('"').strip("'")
    lesson = re.sub(r"^[-*\d.\)\s]+", "", lesson).strip()
    return lesson




def main():
    args = build_argparser("Debate").parse_args()
    run_pipeline(
        load_cases_fn=load_cases,
        distill_fn=distill_lesson,
        model_label="Debate",
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
