from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import torch



SHARED_AGENT_SYSTEM = """You are a trading analyst.

You consider all relevant signals -- chart patterns, momentum (RSI), trend
(MACD), volatility regime, support/resistance levels, candlestick formations,
volume confirmation, and the position of the latest close within the recent
range -- and synthesize them into a single directional decision.

You will receive a sequence of daily bars for a single stock, where t-0 is
the most recent observed day. Each bar contains:
  - Open, High, Low, Close, Volume : raw OHLCV
  - ret_1d    : 1-day log return of Close
  - rsi_14    : 14-day RSI (0..100; >70 overbought, <30 oversold)
  - macd_line : MACD line, EMA12(Close) - EMA26(Close)
  - macd_hist : MACD histogram, macd_line - EMA9(macd_line);
                positive and rising = bullish momentum building,
                negative and falling = bearish momentum building
  - vol_20d   : 20-day annualised realised volatility of ret_1d

Lessons distilled from previous trades:
  0. Do not take a long after a steep multi-day rally with RSI near/above 70 and stretched positive MACD unless price breaks out on expanding volume.


Your task is to DECIDE the action for the stock over the next {HORIZON}
trading day(s): will the Close {HORIZON} trading days from t-0 be higher
(LONG), lower (SHORT), or do the signals not give you enough edge (NO ACTION)?

You MUST respond in this <think>...</think> and <action>...</action> format
with no extra text outside the tags:

<think>
Step-by-step reasoning:
- Summarize what you see in the data (recent trend, RSI, MACD, vol regime).
- List the signals that matter.
- Resolve them into a directional view and a confidence level.
</think>
<action>
A single number -- exactly one of:
   1  => LONG       (Close in {HORIZON} day(s) is expected to be higher)
  -1  => SHORT      (Close in {HORIZON} day(s) is expected to be lower)
   0  => NO ACTION  (insufficient edge / hold)
Do not output any other value. Do not output a magnitude, probability, or log return.
</action>
"""


ORCHESTRATOR_SYSTEM = """You are the head portfolio manager.0. Do not take a long after a steep multi-day rally with RSI near/above 70 and stretched positive MACD unless price breaks out on expanding volume.
  1. When an oversold selloff is followed by RSI reclaiming 50 and MACD histogram turning positive, favor long.
  2. When RSI is sub-40 or rolling down toward/below 50 and MACD histogram is negative or deteriorating during a high-volume breakdown or failed bounce, favor short.

Three independent analysts have each looked at the same window of daily bars
(OHLCV plus RSI, MACD, and 20-day volatility indicators) and produced
(a) a written analysis and (b) a directional vote for the {HORIZON}-day-ahead
move: +1 (long), -1 (short), or 0 (no action). They were given identical
instructions and worked independently; differences between them reflect
genuine uncertainty about the next move.

Lessons distilled from previous trades:
  0. Do not take a long after a steep multi-day rally with RSI near/above 70 and stretched positive MACD unless price breaks out on expanding volume.


Your job is NOT to redo their work. Combine their votes into one decision:
  - When they agree, take the consensus.
  - When they disagree, lean toward the analysis best supported by the data.
  - If conviction is low or signals cancel, output exactly 0 (NO ACTION).

You MUST respond in this <think>...</think> and <action>...</action> format:

<think>
- Note where the three analysts agree and disagree.
- Decide which view dominates and why.
</think>
<action>
A single number -- exactly one of:
   1  => LONG
  -1  => SHORT
   0  => NO ACTION
Do not output any other value.
</action>
"""

class LMixture(torch.nn.Module):
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
        self.save_eval_info = True

        self.horizon = int(getattr(args, "horizon", None)
                           or getattr(config, "horizon", 5))

        self._analyst_system = SHARED_AGENT_SYSTEM.replace(
            "{HORIZON}", str(self.horizon)
        )
        self._orch_system = ORCHESTRATOR_SYSTEM.replace(
            "{HORIZON}", str(self.horizon)
        )

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
                "system": self._analyst_system,
            }
            for i in range(self.num_agents)
        ]

        self.orchestrator = {
            "name": "Head Portfolio Manager",
            "system": self._orch_system,
        }

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

        print(f"[MoA] LLM call failed after {self.max_retries + 1} attempts: {last_err}")
        return "<think>API error; defaulting to no action.</think>\n<action>0.0</action>"

    @staticmethod
    def _format_series(x_single: torch.Tensor) -> str:
        from utils import FEATURE_NAMES
        arr = x_single.detach().cpu().numpy()
        seq_len, n_feat = arr.shape
        assert n_feat == len(FEATURE_NAMES), (
            f"_format_series expects {len(FEATURE_NAMES)} features "
            f"({FEATURE_NAMES}), got {n_feat}."
        )

        widths = [max(10, len(n)) for n in FEATURE_NAMES]
        header = "  t   | " + " | ".join(
            f"{n:>{w}}" for n, w in zip(FEATURE_NAMES, widths)
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for row_idx in range(seq_len):
            days_ago = seq_len - 1 - row_idx
            cells = []
            for col_idx, name in enumerate(FEATURE_NAMES):
                v = float(arr[row_idx, col_idx])
                w = widths[col_idx]
                if name == "Volume":
                    cells.append(f"{v:>{w}.0f}")
                else:
                    cells.append(f"{v:>{w}.4f}")
            lines.append(f" t-{days_ago:02d} | " + " | ".join(cells))
        return "\n".join(lines)

    def _build_data_prompt(self, series_table: str) -> str:
        return (
            "Recent daily bars (OHLCV + indicators) for a single stock; "
            "t-0 is the most recent day:\n\n"
            f"{series_table}\n\n"
            f"Decide the action for this stock over the next {self.horizon} "
            f"trading day(s): LONG (+1), SHORT (-1), or NO ACTION (0). "
            f"Respond strictly in the <think>...</think><action>...</action> format."
        )

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


    def _query_agent(self, agent: Dict[str, Any], data_prompt: str) -> Tuple[str, float]:
        resp = self._call_llm(agent["system"], data_prompt)
        return resp, self._parse_action(resp)

    def _save_trace(
        self,
        x_single: torch.Tensor,
        data_prompt: str,
        agent_outputs: List[Dict[str, Any]],
        orch_user: str,
        orch_resp: str,
        final_decision: float,
        target_return: float | None = None,
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
            "analysts": {
                "system_prompt": self._analyst_system,
                "user_prompt": data_prompt,
                "outputs": [
                    {
                        "name": o["name"],
                        "raw_response": o["response"],
                        "parsed_vote": o["value"],
                    }
                    for o in agent_outputs
                ],
            },
            "orchestrator": {
                "system_prompt": self._orch_system,
                "user_prompt": orch_user,
                "raw_response": orch_resp,
                "parsed_vote": final_decision,
            },
            "final_decision": final_decision,
        }
 
        if self.save_eval_info and target_return is not None:
            hit = None
            if final_decision != 0.0:
                hit = float(np.sign(final_decision) == np.sign(target_return))
            trace["eval"] = {
                "target_return": float(target_return),
                "hit": hit,
            }

        path = os.path.join(self.exp_name, f"data_point_{dp_id:06d}.json")
        try:
            with open(path, "w") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[MoA] failed to write trace {path}: {e}")

    def _process_one(self, x_single: torch.Tensor, target_return: float | None = None) -> float:
        data_prompt = self._build_data_prompt(self._format_series(x_single))
 
        agent_outputs = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.agents))) as ex:
            futs = [ex.submit(self._query_agent, a, data_prompt) for a in self.agents]
            for a, f in zip(self.agents, futs):
                resp, val = f.result()
                agent_outputs.append({"name": a["name"], "response": resp, "value": val})
 
        orch_user = data_prompt + "\n\n--- Independent votes ---\n\n"
        for o in agent_outputs:
            v = o["value"]
            label = "LONG (+1)" if v > 0 else "SHORT (-1)" if v < 0 else "NO ACTION (0)"
            orch_user += (
                f"### {o['name']}\n"
                f"{o['response'].strip()}\n"
                f"(parsed vote: {label})\n\n"
            )
        orch_user += "Now produce the final combined decision in the required format."
 
        orch_resp = self._call_llm(self.orchestrator["system"], orch_user)
        final_decision = self._parse_action(orch_resp)

        self._save_trace(
            x_single, data_prompt, agent_outputs, orch_user, orch_resp, final_decision, target_return=target_return
        )

        return final_decision

    def forward(self, x: torch.Tensor, target_return: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (batch_size, seq_length, num_features)
        returns: (batch_size, 1) tensor of directional decisions in {-1, 0, +1}
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        target_returns = None
        if self.save_eval_info and target_return is not None:
            target_returns = target_return.squeeze(-1) if target_return.dim() > 1 else target_return


        preds = [0.0] * batch_size
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_idx = {
                ex.submit(self._process_one, x[i], float(target_returns[i].item()) if target_returns is not None else None): i for i in range(batch_size)
            }
            for fut in future_to_idx:
                i = future_to_idx[fut]
                try:
                    preds[i] = fut.result()
                except Exception as e:
                    print(f"[MoA] batch item {i} failed: {e}")
                    preds[i] = 0.0

        return torch.tensor(preds, dtype=torch.float32).unsqueeze(-1)