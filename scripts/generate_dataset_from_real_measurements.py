import pandas as pd
import numpy as np
import os
import re
import argparse
from tqdm import tqdm

def create_dataframe_from_folder(folder):
    states = pd.read_csv(os.path.join(folder, "states.csv"))
    inputs = pd.read_csv(os.path.join(folder, "inputs.csv"))

    data = pd.merge(states, inputs, on="ID")

    return data

input_mapping_from_real_to_sim = {
    "phi2": "ux3",
    "phi4": "uy3",
    "phi6": "ux2",
    "phi1": "uy2",
    "phi3": "ux1",
    "phi5": "uy1",
}


def add_virtual_node_at_origin(df):
    def increment_number_in_string(s):
        if not re.search(r'\d+', s):
            return s  # Return unchanged if no number is found
        return re.sub(r'\d+', lambda m: str(int(m.group()) + 1), s, count=1)
    
    # Function must be applied to raw dataset
    for col in sorted(df.columns, key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else float('inf'), reverse=True):
        if col.startswith(("x", "y", "z", "v", "q", "w")):
            new_col = increment_number_in_string(col)
            df[new_col] = df[col]
            df.drop(columns=[col], inplace=True)
    
    for col in ["y1", "vx1", "vy1", "vz1", "qx1", "qy1", "qz1", "w1"]:
        df[col] = 0
    
    df['x1'] = 0.1
    df['z1'] = 0.1 # Will be swapped with y1 later

    return df
            

def rename_columns(data):
    data.rename(columns={col: input_mapping_from_real_to_sim[col] for col in data.columns if col.startswith("phi")}, inplace=True)
    
    data.rename(columns={
        col: "z" + col[1:] if col.startswith("y") else
            "y" + col[1:] if col.startswith("z") else col
        for col in data.columns
    }, inplace=True)

    return data

def process_data(data):
    data = add_virtual_node_at_origin(data)
    # remove columns that are not needed
    data['t'] = data['ID'] * 0.01
    data = rename_columns(data)

    # compute velocity
    for col in data.columns:
        if col.startswith(("x", "y", "z")):
            data[f"v{col}"] = data[col].diff() / 0.01 # use filter insteads

    # compute new states
    for col in data.columns:
        if col.startswith(("x", "y", "z", "v")):
            data[f'{col}_new'] = data[col].shift(-1)

    data = data.iloc[1:-1] # remove first and last row
    data = data[(data['u1'] != 0) | (data['u2'] != 0) | (data['u3'] != 0) | (data['u4'] != 0) | (data['u5'] != 0) | (data['u6'] != 0)]
    data = data.drop(columns=[col for col in data.columns if not col.startswith(("t", "x", "y", "z", "u", "v"))])

    # remove idle states

    data.dropna()

    # Sort columns alphabetically
    data = data[sorted(data.columns)]
    
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
                new_data = new_data.iloc[:500] # keep only 500 rows as in sim datasets
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
    parser.add_argument("--type", type=str, default="train", help="Data type (train, test)")
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
    