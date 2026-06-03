from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

import numpy as np
import torch


GENERATOR_SYSTEM = """You are a trading analyst.

You consider all relevant signals -- chart patterns, momentum (RSI), trend
(MACD), volatility regime, support/resistance levels, candlestick formations,
volume confirmation, and the position of the latest close within the recent
range -- and synthesize them into a single directional decision.

You will receive a sequence of daily bars for a single stock, where t-0 is the
most recent observed day. Each bar contains:
  - Open, High, Low, Close, Volume : raw OHLCV
  - ret_1d    : 1-day log return of Close
  - rsi_14    : 14-day RSI (0..100; >70 overbought, <30 oversold)
  - macd_line : MACD line, EMA12(Close) - EMA26(Close)
  - macd_hist : MACD histogram, macd_line - EMA9(macd_line);
                positive and rising = bullish momentum building,
                negative and falling = bearish momentum building
  - vol_20d   : 20-day annualised realised volatility of ret_1d

Your task is to produce an INITIAL directional view for the {HORIZON}-day-ahead
move: will the Close {HORIZON} trading days from t-0 be higher (LONG), lower
(SHORT), or do the signals not give enough edge (NO ACTION)? A senior reviewer
will audit your reasoning afterwards, so be explicit about which signals you
are relying on.

You MUST respond in this <think>...</think> and <action>...</action> format with no extra text outside the tags:

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


CRITIC_SYSTEM = """You are a risk reviewer.

An analyst has produced an initial directional call on a stock from a recent
window of daily bars (OHLCV plus RSI, MACD, and 20-day volatility indicators)
over a {HORIZON}-day-ahead horizon. You will see both the raw data and the
analyst's full reasoning and vote. Your job is to AUDIT that analysis, not to
redo it.

You produce two things:

1. STRENGTHS. Briefly note what the analyst got right -- signals correctly
   identified, sound inferences, appropriate caution.

2. SERIOUS ISSUES ONLY. Flag problems that, if true, would MATERIALLY change the
   decision. The bar is high. Examples that qualify but not limited to:
     - Misreading the direction of a clear trend.
     - Confusing support and resistance.
     - Ignoring an obvious breakout or breakdown visible in the bars.
     - Citing a chart pattern that is not actually present in the data.
     - Misreading an obvious RSI overbought/oversold signal or MACD crossover.
     - Recommending LONG into a clear downtrend with no reversal signal (or vice versa).
     - Treating noise as signal in a flat, low-conviction tape.
     - Internal contradiction between the stated reasoning and the final vote.

   Do NOT raise:
     - Stylistic concerns.
     - Alternative framings that are merely debatable.
     - Generic disclaimers ("past performance does not guarantee ...").
     - Issues you cannot point to in the actual data.

   If the analyst's call is reasonable -- even if not the one you would have made --
   say so plainly and do NOT invent issues to look thorough.

You do NOT cast a vote. You produce a structured review.

You MUST respond in this format with no extra text outside the tags:

<strengths>
- Bullet points of what the analyst handled correctly.
</strengths>
<serious_issues>
- Bullet points of material problems that should change the decision.
- If there are no serious issues, write exactly: NONE
</serious_issues>
"""


REFINER_SYSTEM = """You are the head portfolio manager. You make the final call.

You will receive:
  (a) the window of daily bars (OHLCV + RSI/MACD/vol indicators) for a stock,
  (b) an initial directional call from a junior analyst for the {HORIZON}-day-ahead
      move (full reasoning + vote),
  (c) a senior reviewer's audit of that call, listing strengths and any serious issues.

Your job is to produce the final decision. Guidelines:

  - If the reviewer found NO serious issues, the initial call should stand.
  - If the reviewer flagged genuine serious issues, override or downgrade the
    initial call accordingly -- including going to NO ACTION (0) if the original
    reasoning collapses, or flipping the sign if the data actually points the
    other way.
  - Do not invent new objections the reviewer did not raise.
  - Do not treat "the reviewer mentioned something" as automatically requiring a
    reversal -- weigh whether each issue actually undermines the conclusion.
  - If conviction is low after weighing both sides, output exactly 0.

You MUST respond in this <think>...</think> and <action>...</action> format:

<think>
- State whether you are upholding, downgrading, or reversing the initial call.
- Reference the strengths and the serious issues you weighed.
</think>
<action>
A single number -- exactly one of:
   1  => LONG
  -1  => SHORT
   0  => NO ACTION
Do not output any other value.
</action>
"""


class Sequential(torch.nn.Module):
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
        self.save_eval_info = True

        self.horizon = int(getattr(args, "horizon", None)
                           or getattr(config, "horizon", 5))

        self._gen_system     = GENERATOR_SYSTEM.replace("{HORIZON}", str(self.horizon))
        self._critic_system  = CRITIC_SYSTEM.replace("{HORIZON}", str(self.horizon))
        self._refiner_system = REFINER_SYSTEM.replace("{HORIZON}", str(self.horizon))

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

        print(f"[Sequential] LLM call failed after {self.max_retries + 1} attempts: {last_err}")
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
            f"trading day(s): LONG (+1), SHORT (-1), or NO ACTION (0)."
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

    @staticmethod
    def _parse_critic(response: str) -> Dict[str, Any]:
        if not response:
            return {"strengths": "", "serious_issues": "", "has_serious_issues": False}

        s = re.search(r"<strengths>\s*(.*?)\s*</strengths>", response, re.DOTALL | re.IGNORECASE)
        i = re.search(r"<serious_issues>\s*(.*?)\s*</serious_issues>", response, re.DOTALL | re.IGNORECASE)
        strengths = s.group(1).strip() if s else ""
        issues = i.group(1).strip() if i else ""

        norm = re.sub(r"[\s\-\*\.]+", "", issues).lower()
        has_serious = bool(issues) and norm != "none"

        return {
            "strengths": strengths,
            "serious_issues": issues,
            "has_serious_issues": has_serious,
        }

    @staticmethod
    def _vote_label(v: float) -> str:
        if v > 0:
            return "LONG (+1)"
        if v < 0:
            return "SHORT (-1)"
        return "NO ACTION (0)"

    def _save_trace(
        self,
        x_single: torch.Tensor,
        data_prompt: str,
        gen_resp: str,
        gen_vote: float,
        critic_user: str,
        critic_resp: str,
        critic_parsed: Dict[str, Any],
        refiner_user: str,
        refiner_resp: str,
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
            "pipeline": "sequential",
            "input": {
                "ohlcv_window": x_single.detach().cpu().tolist(),
                "formatted_series": self._format_series(x_single),
            },
            "generator": {
                "system_prompt": self._gen_system,
                "user_prompt": data_prompt,
                "raw_response": gen_resp,
                "parsed_vote": gen_vote,
            },
            "critic": {
                "system_prompt": self._critic_system,
                "user_prompt": critic_user,
                "raw_response": critic_resp,
                "parsed": critic_parsed,
            },
            "refiner": {
                "system_prompt": self._refiner_system,
                "user_prompt": refiner_user,
                "raw_response": refiner_resp,
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
            print(f"[Sequential] failed to write trace {path}: {e}")

    def _process_one(self, x_single: torch.Tensor,
                     target_return: float | None = None) -> float:
        data_prompt = self._build_data_prompt(self._format_series(x_single))

        gen_resp = self._call_llm(self._gen_system, data_prompt)
        gen_vote = self._parse_action(gen_resp)

        critic_user = (
            f"{data_prompt}\n\n"
            "--- Initial analyst output ---\n\n"
            f"{gen_resp.strip()}\n\n"
            f"(parsed vote: {self._vote_label(gen_vote)})\n\n"
            "Now audit the analyst's reasoning in the required "
            "<strengths>/<serious_issues> format."
        )
        critic_resp = self._call_llm(self._critic_system, critic_user)
        critic_parsed = self._parse_critic(critic_resp)

        refiner_user = (
            f"{data_prompt}\n\n"
            "--- Initial analyst output ---\n\n"
            f"{gen_resp.strip()}\n\n"
            f"(parsed vote: {self._vote_label(gen_vote)})\n\n"
            "--- Senior reviewer's audit ---\n\n"
            f"{critic_resp.strip()}\n\n"
            "Now produce the final combined decision in the required format."
        )
        refiner_resp = self._call_llm(self._refiner_system, refiner_user)
        final_decision = self._parse_action(refiner_resp)

        self._save_trace(
            x_single=x_single,
            data_prompt=data_prompt,
            gen_resp=gen_resp,
            gen_vote=gen_vote,
            critic_user=critic_user,
            critic_resp=critic_resp,
            critic_parsed=critic_parsed,
            refiner_user=refiner_user,
            refiner_resp=refiner_resp,
            final_decision=final_decision,
            target_return=target_return,
        )

        return final_decision

    def forward(self, x: torch.Tensor,
                target_return: torch.Tensor | None = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size = x.shape[0]

        target_returns = None
        if self.save_eval_info and target_return is not None:
            target_returns = (target_return.squeeze(-1)
                              if target_return.dim() > 1 else target_return)

        preds = [0.0] * batch_size
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            future_to_idx = {
                ex.submit(
                    self._process_one, x[i],
                    float(target_returns[i].item())
                    if target_returns is not None else None,
                ): i for i in range(batch_size)
            }
            for fut in future_to_idx:
                i = future_to_idx[fut]
                try:
                    preds[i] = fut.result()
                except Exception as e:
                    print(f"[Sequential] batch item {i} failed: {e}")
                    preds[i] = 0.0

        return torch.tensor(preds, dtype=torch.float32).unsqueeze(-1)