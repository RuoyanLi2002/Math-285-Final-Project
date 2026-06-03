import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


FEATURE_NAMES = [
    "Open", "High", "Low", "Close", "Volume",
    "ret_1d", "rsi_14", "macd_line", "macd_hist", "vol_20d",
]
CLOSE_IDX = 3
NUM_FEATURES = len(FEATURE_NAMES)



def impute_missing_values(data_array):
    num_stocks, num_time_steps, num_features = data_array.shape
    cleaned = data_array.copy()
    for i in range(num_stocks):
        stock_data = cleaned[i]
        if np.isnan(stock_data).any():
            df = pd.DataFrame(stock_data)
            df = df.interpolate(method="linear", limit_direction="both", axis=0)
            df = df.ffill(axis=0).bfill(axis=0)
            cleaned[i] = df.values
    if np.isnan(cleaned).any():
        print("Warning: NaNs still exist! Filling remaining with 0.")
        cleaned = np.nan_to_num(cleaned)
    return cleaned


def _ema(x: np.ndarray, span: int) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(x, dtype=np.float64)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, period)
    avg_loss = _ema(loss, period)
    rs = avg_gain / np.maximum(avg_loss, 1e-12)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return np.clip(rsi, 0.0, 100.0)


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = _ema(close, fast) - _ema(close, slow)
    macd_signal = _ema(macd_line, signal)
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist


def _rolling_vol(returns: np.ndarray, window: int = 20,
                 annualize: bool = True) -> np.ndarray:
    out = np.zeros_like(returns, dtype=np.float64)
    s1 = 0.0
    s2 = 0.0
    for i, r in enumerate(returns):
        s1 += r
        s2 += r * r
        if i >= window:
            old = returns[i - window]
            s1 -= old
            s2 -= old * old
        n = min(i + 1, window)
        if n > 1:
            mean = s1 / n
            var = max(s2 / n - mean * mean, 0.0)
            out[i] = np.sqrt(var)
    if annualize:
        out = out * np.sqrt(252.0)
    return out


def compute_indicators(ohlcv: np.ndarray) -> np.ndarray:
    assert ohlcv.shape[1] == 5, f"expected 5 OHLCV cols, got {ohlcv.shape[1]}"
    close = ohlcv[:, CLOSE_IDX].astype(np.float64)
    safe_close = np.maximum(close, 1e-12)

    ret_1d = np.zeros_like(close)
    ret_1d[1:] = np.log(safe_close[1:] / safe_close[:-1])

    rsi_14 = _rsi(safe_close, period=14)
    macd_line, _, macd_hist = _macd(safe_close, fast=12, slow=26, signal=9)
    vol_20d = _rolling_vol(ret_1d, window=20, annualize=True)

    extras = np.stack([ret_1d, rsi_14, macd_line, macd_hist, vol_20d], axis=1)
    out = np.concatenate([ohlcv, extras], axis=1).astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out



def select_liquid_stocks(data_array: np.ndarray,
                         sorted_permnos,
                         top_n: int):
    if top_n <= 0 or top_n >= len(sorted_permnos):
        return data_array, list(sorted_permnos)
    close = data_array[:, :, CLOSE_IDX]
    volume = data_array[:, :, 4]
    dollar_vol = np.nanmean(close * volume, axis=1)
    top_idx = np.argsort(-dollar_vol)[:top_n]
    selected = data_array[top_idx]
    selected_permnos = [sorted_permnos[i] for i in top_idx]
    print(f"Selected top {top_n} liquid stocks (PERMNOs): {selected_permnos}")
    print(f"  mean dollar volumes: "
          f"{[float(dollar_vol[i]) for i in top_idx]}")
    return selected, selected_permnos



def load_dataset(args, start_year=2015, end_year=2020, shuffle=True):
    horizon = int(getattr(args, "horizon", 1) or 1)
    top_n = int(getattr(args, "top_n_stocks", 0) or 0)
    burn_in = int(getattr(args, "burn_in", 30) or 0)


    all_files = []
    for year in range(start_year, end_year + 1):
        pattern = os.path.join(args.dataset_root, str(year), "*.csv.gz")
        all_files.extend(glob.glob(pattern))
    sorted_files = sorted(all_files)
    print(f"Found {len(sorted_files)} files.")

    common_permnos = None
    for i, f in enumerate(sorted_files):
        if i % 100 == 0:
            print(f"Scanning file {i}/{len(sorted_files)}...")
        try:
            df = pd.read_csv(f, usecols=["PERMNO"])
            permnos = set(df["PERMNO"].unique())
            common_permnos = permnos if common_permnos is None \
                else common_permnos.intersection(permnos)
            if not common_permnos:
                print("Intersection empty.")
                return None
        except Exception as e:
            print(f"Error reading {f}: {e}")

    sorted_permnos = sorted(common_permnos)
    permno_to_idx = {p: i for i, p in enumerate(sorted_permnos)}
    num_stocks = len(sorted_permnos)
    num_time_steps = len(sorted_files)
    print(f"Total continuously active stocks: {num_stocks}")
    print(f"Total time steps: {num_time_steps}")


    raw = np.zeros((num_stocks, num_time_steps, 5), dtype=np.float64)
    for t, f in enumerate(sorted_files):
        if t % 100 == 0:
            print(f"Processing {t}/{len(sorted_files)}...")
        try:
            df = pd.read_csv(f)
            df = df[df["PERMNO"].isin(permno_to_idx)]
            row_indices = df["PERMNO"].map(permno_to_idx).values
            raw[row_indices, t, :] = df[
                ["open", "high", "low", "close", "volume"]
            ].values
        except Exception as e:
            print(f"Error processing {f}: {e}")

    raw = impute_missing_values(raw)
    print(f"raw OHLCV array: {raw.shape}")


    raw, sorted_permnos = select_liquid_stocks(raw, sorted_permnos, top_n)


    enriched = np.zeros(
        (raw.shape[0], raw.shape[1], NUM_FEATURES), dtype=np.float32
    )
    for i in range(raw.shape[0]):
        enriched[i] = compute_indicators(raw[i])
    print(f"enriched feature array: {enriched.shape}  "
          f"(features = {FEATURE_NAMES})")

    
    
    x_list, y_list = [], []
    for s in range(enriched.shape[0]):
        stock_data = enriched[s]
        end_offset = args.seq_length + horizon - 1
        for t in range(burn_in,
                       num_time_steps - end_offset,
                       args.split_interval):
            x_window = stock_data[t : t + args.seq_length, :]
            y_target = stock_data[t + end_offset, :]
            x_list.append(x_window)
            y_list.append(y_target)

    x_np = np.asarray(x_list, dtype=np.float32)
    y_np = np.asarray(y_list, dtype=np.float32)
    print(f"Final X shape: {x_np.shape}   "
          f"(horizon={horizon}, burn_in={burn_in})")
    print(f"Final y shape: {y_np.shape}")

    dataset = TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np))
    return DataLoader(dataset, batch_size=1, shuffle=shuffle)


def load_train(args):
    file_path = f"{args.data_save_path}/train.pth"
    if os.path.exists(file_path):
        print(f"train.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        return DataLoader(all_data, batch_size=1, shuffle=True)
    print("train.pth does not exist. Create dataset")
    return load_dataset(args)