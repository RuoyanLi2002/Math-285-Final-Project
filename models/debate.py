from __future__ import annotations

import json
import os
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


DEBATE_AGENT_SYSTEM = """You are a trading analyst participating in a multi-round debate.

You consider all relevant signals -- chart patterns (double tops/bottoms,
head-and-shoulders, breakouts), support/resistance levels, candlestick formations,
momentum, volume confirmation, trend vs mean-reversion regime, dispersion of
returns, and the position of the latest close within the recent range -- and
synthesize them into a single directional decision.

You will receive a sequence of daily OHLCV bars (open, high, low, close, volume)
for a single stock, where t-0 is the most recent observed day.

In round 1 you analyze the data independently. In later rounds you will also see
the previous-round analyses from the other analysts (and your own). You may
incorporate their strongest points, push back on weak reasoning, or stand your
ground -- but think for yourself. Do not change your vote just to match the
group; only change it if the evidence or another analyst's argument actually
moves you.

Your task is to DECIDE the next-day action for the stock.

You MUST respond in this <think>...</think> and <action>...</action> format with no extra text outside the tags:

<think>
Step-by-step reasoning:
- Summarize what you see in the data.
- (Round 2+) State where you agree or disagree with the other analysts and why.
- List the signals that matter.
- Resolve them into a directional view and a confidence level.
</think>
<action>
A single number -- exactly one of:
   1  => LONG       (close is expected to rise)
  -1  => SHORT      (close is expected to fall)
   0  => NO ACTION  (insufficient edge / hold)
Do not output any other value. Do not output a magnitude, probability, or log return.
</action>
"""


class Debate(torch.nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.config = config

        model_field = getattr(config, "base_model", None)
        if not model_field:
            raise ValueError(
                "config.base_model must be set, e.g. 'openai:gpt-4o-mini' or "
                "'gemini:gemini-2.0-flash'."
            )

        if ":" in model_field:
            provider, model_name = model_field.split(":", 1)
        else:
            mf = model_field.lower()
            if "gemini" in mf:
                provider, model_name = "gemini", model_field
            elif mf.startswith(("gpt", "o1", "o3", "o4")):
                provider, model_name = "openai", model_field
            else:
                raise ValueError(
                    f"Cannot infer provider from model='{model_field}'. "
                    "Use 'openai:<model>' or 'gemini:<model>'."
                )

        self.provider = provider.lower().strip()
        self.model_name = model_name.strip()

        self.api_key = getattr(config, "api_key", None)
        if not self.api_key:
            if self.provider == "openai":
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.provider == "gemini":
                self.api_key = (
                    os.environ.get("GEMINI_API_KEY")
                    or os.environ.get("GOOGLE_API_KEY")
                )
        if not self.api_key:
            raise ValueError(f"No API key found for provider '{self.provider}'.")

        self.temperature = float(getattr(config, "temperature", 0.3))
        self.max_workers = int(getattr(config, "max_workers", 4))
        self.max_retries = int(getattr(config, "max_retries", 2))
        self.num_agents = int(getattr(config, "num_agents", 3))
        self.num_rounds = max(1, int(getattr(config, "num_rounds", 2)))

        self.exp_name = args.exp_name
        if self.exp_name:
            if not os.path.exists(self.exp_name):
                os.makedirs(self.exp_name)
                print(f"Folder '{self.exp_name}' created.")
            else:
                print(f"Folder '{self.exp_name}' already exists.")
        self._save_counter = 0
        self._save_lock = threading.Lock()

        self._init_client()

        self.agents = [
            {
                "key": f"analyst_{i + 1}",
                "name": f"Analyst {i + 1}",
                "system": DEBATE_AGENT_SYSTEM,
            }
            for i in range(self.num_agents)
        ]

    def _init_client(self):
        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError("Install with: pip install openai") from e
            self.client = OpenAI(api_key=self.api_key)

        elif self.provider == "gemini":
            try:
                from google import genai
            except ImportError as e:
                raise ImportError("Install with: pip install google-genai") from e
            self.client = genai.Client(api_key=self.api_key)

        else:
            raise ValueError(
                f"Unsupported provider: '{self.provider}'. Use 'openai' or 'gemini'."
            )

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        last_err = None
        for _ in range(self.max_retries + 1):
            try:
                if self.provider == "openai":
                    resp = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.temperature,
                    )
                    return resp.choices[0].message.content or ""

                elif self.provider == "gemini":
                    from google.genai import types
                    resp = self.client.models.generate_content(
                        model=self.model_name,
                        contents=user_prompt,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=self.temperature,
                        ),
                    )
                    return resp.text or ""
            except Exception as e:
                last_err = e

        print(f"[Debate] LLM call failed after {self.max_retries + 1} attempts: {last_err}")
        return "<think>API error; defaulting to no action.</think>\n<action>0.0</action>"

    @staticmethod
    def _format_series(x_single: torch.Tensor) -> str:
        arr = x_single.detach().cpu().numpy()
        seq_len, n_feat = arr.shape
        assert n_feat == 5, (
            f"_format_series expects 5 features (OHLCV), got {n_feat}."
        )

        names = ["Open", "High", "Low", "Close", "Volume"]
        header = "  t   | " + " | ".join(f"{n:>10}" for n in names)
        sep = "-" * len(header)
        lines = [header, sep]

        for row_idx in range(seq_len):
            days_ago = seq_len - 1 - row_idx  # row 0 -> oldest, last row -> t-0
            cells = []
            for col_idx in range(n_feat):
                v = float(arr[row_idx, col_idx])
                if col_idx == 4:  # Volume
                    cells.append(f"{v:>10.0f}")
                else:  # OHLC prices
                    cells.append(f"{v:>10.4f}")
            lines.append(f" t-{days_ago:02d} | " + " | ".join(cells))
        return "\n".join(lines)

    @staticmethod
    def _build_data_prompt(series_table: str) -> str:
        return (
            "Recent daily OHLCV window for a single stock (t-0 is the most recent day):\n\n"
            f"{series_table}\n\n"
            "Decide the next-day action for this stock: LONG (+1), SHORT (-1), or "
            "NO ACTION (0). Respond strictly in the "
            "<think>...</think><action>...</action> format."
        )

    @staticmethod
    def _vote_label(v: float) -> str:
        if v > 0:
            return "LONG (+1)"
        if v < 0:
            return "SHORT (-1)"
        return "NO ACTION (0)"

    def _build_debate_prompt(
        self,
        base_prompt: str,
        prev_round_outputs: List[Dict[str, Any]],
        round_idx: int,
        self_name: str,
    ) -> str:
        """User prompt for round 2+; shows every agent's previous-round response."""
        parts = [base_prompt, "", f"--- Round {round_idx - 1} analyses ---", ""]
        for o in prev_round_outputs:
            tag = " (you)" if o["name"] == self_name else ""
            parts.append(f"### {o['name']}{tag}")
            parts.append(o["response"].strip())
            parts.append(f"(parsed vote: {self._vote_label(o['value'])})")
            parts.append("")
        parts.append(
            f"This is round {round_idx} of {self.num_rounds}. Reconsider your view in "
            "light of the analyses above. You may keep or change your vote; only "
            "change it if the evidence or another analyst's argument actually moves "
            "you. Respond strictly in the <think>...</think><action>...</action> format."
        )
        return "\n".join(parts)

    @staticmethod
    def _parse_action(response: str) -> float:
        if not response:
            return 0.0

        m = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL | re.IGNORECASE)
        body = m.group(1) if m else response
        low = body.lower()

        has_long = bool(re.search(r"\blong\b", low))
        has_short = bool(re.search(r"\bshort\b", low))
        has_hold = bool(re.search(r"\b(hold|do nothing|no action|no[- ]op|neutral)\b", low))

        nums = re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", body)
        if nums:
            try:
                val = float(nums[0])
            except ValueError:
                val = 0.0

            if has_short and val > 0:
                val = -1.0
            elif has_long and val < 0:
                val = 1.0

            if val > 0:
                return 1.0
            if val < 0:
                return -1.0

            if has_long:
                return 1.0
            if has_short:
                return -1.0
            return 0.0

        if has_long:
            return 1.0
        if has_short:
            return -1.0
        if has_hold:
            return 0.0
        return 0.0

    def _query_agent(
        self, agent: Dict[str, Any], user_prompt: str
    ) -> Tuple[str, float]:
        resp = self._call_llm(agent["system"], user_prompt)
        return resp, self._parse_action(resp)

    def _run_round(
        self,
        round_user_prompts: List[str],
    ) -> List[Dict[str, Any]]:
        """Run one debate round across all agents in parallel. Each agent gets its
        own user prompt (the per-agent debate prompt), and the system prompt is
        shared from agent['system']."""
        outputs: List[Dict[str, Any]] = [None] * len(self.agents)  # type: ignore
        with ThreadPoolExecutor(max_workers=max(1, len(self.agents))) as ex:
            futs = {
                ex.submit(self._query_agent, a, round_user_prompts[i]): i
                for i, a in enumerate(self.agents)
            }
            for fut, i in futs.items():
                resp, val = fut.result()
                outputs[i] = {
                    "name": self.agents[i]["name"],
                    "response": resp,
                    "value": val,
                }
        return outputs

    @staticmethod
    def _majority_vote(votes: List[float]) -> float:
        """Majority vote across {-1, 0, +1}. Ties resolve to 0 (no action) -- the
        conservative posture used elsewhere in this codebase."""
        if not votes:
            return 0.0
        counts = Counter(round(v) for v in votes)
        max_count = max(counts.values())
        winners = [v for v, c in counts.items() if c == max_count]
        if len(winners) == 1:
            return float(winners[0])
        # tie -- prefer 0 if it's among the winners; otherwise stay flat
        if 0 in winners:
            return 0.0
        return 0.0

    def _save_trace(
        self,
        x_single: torch.Tensor,
        data_prompt: str,
        rounds: List[List[Dict[str, Any]]],
        final_decision: float,
    ) -> None:
        if not self.exp_name:
            return

        with self._save_lock:
            dp_id = self._save_counter
            self._save_counter += 1

        trace = {
            "data_point_id": dp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "input": {
                "ohlcv_window": x_single.detach().cpu().tolist(),
                "formatted_series": self._format_series(x_single),
            },
            "config": {
                "num_agents": self.num_agents,
                "num_rounds": self.num_rounds,
            },
            "analysts": {
                "system_prompt": DEBATE_AGENT_SYSTEM,
                "round_1_user_prompt": data_prompt,
                "rounds": [
                    {
                        "round": r_idx + 1,
                        "outputs": [
                            {
                                "name": o["name"],
                                "raw_response": o["response"],
                                "parsed_vote": o["value"],
                            }
                            for o in round_outputs
                        ],
                    }
                    for r_idx, round_outputs in enumerate(rounds)
                ],
            },
            "voting": {
                "method": "majority_vote",
                "tie_break": "no_action",
                "final_round_votes": [o["value"] for o in rounds[-1]],
            },
            "final_decision": final_decision,
        }

        path = os.path.join(self.exp_name, f"data_point_{dp_id:06d}.json")
        try:
            with open(path, "w") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[Debate] failed to write trace {path}: {e}")

    def _process_one(self, x_single: torch.Tensor) -> float:
        data_prompt = self._build_data_prompt(self._format_series(x_single))

        rounds: List[List[Dict[str, Any]]] = []

        # Round 1: every agent gets the same independent prompt.
        round_prompts = [data_prompt for _ in self.agents]
        first = self._run_round(round_prompts)
        rounds.append(first)

        # Rounds 2..N: every agent sees the previous round's outputs from every
        # agent (including its own) and is asked to reconsider.
        for r in range(2, self.num_rounds + 1):
            prev = rounds[-1]
            round_prompts = [
                self._build_debate_prompt(
                    base_prompt=data_prompt,
                    prev_round_outputs=prev,
                    round_idx=r,
                    self_name=a["name"],
                )
                for a in self.agents
            ]
            rounds.append(self._run_round(round_prompts))

        final_votes = [o["value"] for o in rounds[-1]]
        final_decision = self._majority_vote(final_votes)

        self._save_trace(x_single, data_prompt, rounds, final_decision)

        return final_decision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_length, num_features)
        returns: (batch_size, 1) tensor of directional decisions in {-1, 0, +1}
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        preds = [0.0] * batch_size
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_idx = {
                ex.submit(self._process_one, x[i]): i for i in range(batch_size)
            }
            for fut in future_to_idx:
                i = future_to_idx[fut]
                try:
                    preds[i] = fut.result()
                except Exception as e:
                    print(f"[Debate] batch item {i} failed: {e}")
                    preds[i] = 0.0

        return torch.tensor(preds, dtype=torch.float32).unsqueeze(-1)