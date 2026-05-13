import os
import re
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


import numpy as np
import pandas as pd

def impute_missing_values(data_array):
    num_stocks, num_time_steps, num_features = data_array.shape
    
    cleaned_array = data_array.copy()

    for i in range(num_stocks):
        stock_data = cleaned_array[i]
        
        if np.isnan(stock_data).any():
            df = pd.DataFrame(stock_data)
            
            df = df.interpolate(method='linear', limit_direction='both', axis=0)
            df = df.ffill(axis=0).bfill(axis=0)
            
            # df = df.fillna(0.0)
            
            cleaned_array[i] = df.values
            
    if np.isnan(cleaned_array).any():
        print("Warning: NaNs still exist! Filling remaining with 0.")
        cleaned_array = np.nan_to_num(cleaned_array)
        
    return cleaned_array


def load_dataset(args, start_year=2015, end_year=2020, shuffle=True):
    all_files = []
    for year in range(start_year, end_year + 1):
        pattern = os.path.join(args.dataset_root, str(year), '*.csv.gz')
        files = glob.glob(pattern)
        all_files.extend(files)
    
    # Sort by filename (ensures chronological order YYYYMMDD)
    sorted_files = sorted(all_files)

    print(f"Found {sorted_files}")
    
    common_permnos = None
    
    for i, f in enumerate(sorted_files):
        if i % 100 == 0:
            print(f"Scanning file {i}/{len(sorted_files)}...")
        
        try:
            df = pd.read_csv(f, usecols=['PERMNO'])
            current_permnos = set(df['PERMNO'].unique())
            
            if common_permnos is None:
                common_permnos = current_permnos
            else:
                common_permnos = common_permnos.intersection(current_permnos)

            if not common_permnos:
                print("Intersection is empty! No stocks exist for the entire duration.")
                return None, None, None
        except Exception as e:
            print(f"Error reading {f}: {e}")

    sorted_permnos = sorted(list(common_permnos))
    permno_to_idx = {p: i for i, p in enumerate(sorted_permnos)}
    
    num_stocks = len(sorted_permnos)
    num_time_steps = len(sorted_files)
    
    print(f"Total continuously active stocks: {num_stocks}")
    print(f"Total time steps: {num_time_steps}")

    data_array = np.zeros((num_stocks, num_time_steps, 5))

    for t, f in enumerate(sorted_files):
        if t % 100 == 0:
            print(f"Processing {t}/{len(sorted_files)}...")
            
        try:
            df = pd.read_csv(f)
            df = df[df['PERMNO'].isin(permno_to_idx)]
            
            row_indices = df['PERMNO'].map(permno_to_idx).values
            values = df[['open', 'high', 'low', 'close', 'volume']].values
            # values = df[['open', 'high', 'low', 'close']].values
            
            data_array[row_indices, t, :] = values
        except Exception as e:
            print(f"Error processing {f}: {e}")

    data_array = impute_missing_values(data_array)
    print(f"data_array: {data_array.shape}")
    

    x_list = []
    y_list = []

    for stock_idx in range(num_stocks):
        if stock_idx % 100 == 0:
            print(f"Processing stock {stock_idx}/{num_stocks}...")
        
        stock_data = data_array[stock_idx]
        for t in range(0, num_time_steps - args.seq_length, args.split_interval):
            t_end = t + args.seq_length
            x_window = stock_data[t : t_end, :]
            y_target = stock_data[t_end, :]
            
            x_list.append(x_window)
            y_list.append(y_target)

    x_np = np.array(x_list, dtype=np.float32)
    y_np = np.array(y_list, dtype=np.float32)
    
    tensor_x = torch.from_numpy(x_np)
    tensor_y = torch.from_numpy(y_np)
    
    print(f"Final X Shape: {tensor_x.shape}")
    print(f"Final y Shape: {tensor_y.shape}")
    
    # Create Dataset and DataLoader
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    return dataloader





def load_train(args):
    file_path = f"{args.data_save_path}/train.pth"

    if os.path.exists(file_path):
        print(f"train.pth already exists. Load from {file_path}")
        all_data = torch.load(file_path)
        print(len(all_data))
        dataloader = DataLoader(all_data, batch_size=1, shuffle=True)
    else:
        print(f"train.pth does not exists. Create dataset")
        dataloader = load_dataset(args)

    return dataloader