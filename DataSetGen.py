from utils_sionna.utils import preprocess_complex_csi, cell_ids_2_positions, position_2_cell_idx,\
    gen_CSI_matrix_and_scatter_position_matrix
import numpy as np
import os
import json
import time
import random
import yaml
import re

def is_config_empty(config_data):
    return not bool(config_data)

def read_config(file_path):
    """Read configuration from a JSON or YAML file."""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif file_path.endswith((".yml", ".yaml")):
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def extract_number(filename):
    """extract the first number found in the filename for sorting."""
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else 0


def main(config_folder, output_path):
    all_results = []
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    config_files = [
        f for f in os.listdir(config_folder)
        if f.endswith((".json", ".yml", ".yaml"))
    ]
    config_files.sort(key=extract_number)

    for filename in config_files:
        file_path = os.path.join(config_folder, filename)

        try:
            config = read_config(file_path)
            if is_config_empty(config):
                print(f"⚠️ skip empty config file: {filename}")
                continue

            print(f"✅ Simulating: {filename}")
            result = gen_CSI_matrix_and_scatter_position_matrix(config)
            all_results.append(result)

        except Exception as e:
            print(f"❌ Fail to read or simulate {filename} : {e}")

    # 保存结果
    np.save(output_path, all_results)
    print(f"\n✅ All completed, save to {output_path}")


if __name__ == "__main__":
    config_folder = "./configs"         
    output_path = "./dataset/sim_results.npy"  # save path
    main(config_folder, output_path)


