import os
import numpy as np
import torch
from tqdm import tqdm

from utils import load_dataset, CLOSE_IDX


def eval(args, model):
    horizon = int(getattr(args, "horizon", 1) or 1)
    test_loader = load_dataset(args, start_year=2021, end_year=2021, shuffle=True)
    all_strategy_returns = []

    hit_rate = 0.0
    i = 0
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Eval", total=len(test_loader), unit="batch"):
            close_t        = x[:, -1, CLOSE_IDX]
            close_t_plus_h = y[:, CLOSE_IDX]
            target_return  = torch.log(close_t_plus_h / close_t).unsqueeze(-1)

            pred = model(x, target_return)

            mask = pred.squeeze(-1) != 0
            if mask.any():
                hit_rate += (
                    torch.sign(pred.squeeze(-1)[mask])
                    == torch.sign(target_return.squeeze(-1)[mask])
                ).float().mean()

            strategy_returns = torch.sign(pred.squeeze(-1)) * target_return.squeeze(-1)
            all_strategy_returns.append(strategy_returns.detach().cpu())

    hit_rate = float(hit_rate / max(i, 1))
    pnl = float("nan")
    sharpe = float("nan")
    if all_strategy_returns:
        sr = torch.cat(all_strategy_returns).numpy()
        pnl = float(sr.sum())
        if sr.std() > 0:
            scale = np.sqrt(252.0 / max(horizon, 1))
            sharpe = float(sr.mean() / (sr.std() + 1e-8) * scale)

    tag = f"{horizon}-day-ahead"
    print(f"Simple long/short Hit Rate on 2021 ({tag}): {hit_rate:.6f}")
    print(f"Simple long/short PnL on 2021 ({tag}): {pnl:.6f}")
    print(f"Simple long/short Sharpe on 2021 ({tag}): {sharpe:.4f}")

    results_path = os.path.join(args.exp_name, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Simple long/short Hit Rate on 2021 ({tag}): {hit_rate:.6f}\n")
        f.write(f"Simple long/short PnL on 2021 ({tag}): {pnl:.6f}\n")
        f.write(f"Simple long/short Sharpe on 2021 ({tag}): {sharpe:.4f}\n")