import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm

def create_dataframe_from_folder(folder):
    states = pd.read_csv(os.path.join(folder, "states.csv"))
    inputs = pd.read_csv(os.path.join(folder, "inputs.csv"))

    data = pd.merge(states, inputs, on="ID")

    return data

def process_data(data):
    # remove columns that are not needed
    data['t'] = data['ID'] * 0.01
    data = data.drop(columns=[col for col in data.columns if not col.startswith(("t", "x", "y", "z", "u"))])

    # compute velocity
    for col in data.columns:
        if col.startswith(("x", "y", "z")):
            data[f"v{col}"] = data[col].diff() / 0.01 # use filter insteads

    # compute new states
    for col in data.columns:
        data[f'{col}_new'] = data[col].shift(-1)

    data = data.iloc[:-1] # remove last row
    data.dropna()

    # remove idle states
    data = data[data['u1'] != 0] 

    return data

def split_test_data(data):
    segments = []
    current_segment = []
    count = 0

    num_test_files = 0
    for _, row in data.iterrows():
        if count > 1000:
            if row['u1'] == 0 and row['u2'] == 0 and row['u3'] == 0 and row['u4'] == 0 and row['u5'] == 0 and row['u6'] == 0:
                new_data = pd.DataFrame(current_segment)
                new_data = new_data[(new_data['u1'] != 0) | (new_data['u2'] != 0) | (new_data['u3'] != 0) | (new_data['u4'] != 0) | (new_data['u5'] != 0) | (new_data['u6'] != 0)]
                segments.append(new_data)

                print(f"Segment #{num_test_files} with {len(segments[-1])} rows")
                num_test_files, count, current_segment = num_test_files + 1, 0, []

        current_segment.append(row) 
        count += 1
        
        if num_test_files == 10:
            break

    return segments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset from real measurements")
    parser.add_argument("--folder", type=str, default="data/long_rollouts_100_max_amplitude_80_policy_harmonic_inputs", help="Folder containing the data")
    parser.add_argument("--type", type=str, default="test", help="Data type (train, test)")
    args = parser.parse_args()

    folder = os.path.join(args.folder, args.type)
    if not os.path.exists(folder):
        raise ValueError(f"Folder {folder} does not exist")

    data = create_dataframe_from_folder(folder)

    if args.type == "test":
        data_segments = split_test_data(data)
        for i, data_segment in enumerate(tqdm(data_segments)):
            os.makedirs(os.path.join(folder, str(i+1), "raw"), exist_ok=True)
            process_data(data_segment).to_csv(os.path.join(folder, str(i+1), "raw", "data.csv"), index=False)
    else:
        os.makedirs(os.path.join(folder, "raw"), exist_ok=True)
        process_data(data).to_csv(os.path.join(folder, "raw", "data.csv"), index=False)

    print(f"Data saved!")
    