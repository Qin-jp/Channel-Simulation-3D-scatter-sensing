from utils_sionna.ChannelDataGen import gen_CSI_matrix_and_scatter_position_matrix
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
            num_data = 0
            while True:              
                print(f"✅ Simulating: {filename}")
                result = gen_CSI_matrix_and_scatter_position_matrix(config)
                num_data += len(result["CSI"])
                print(f"   Current total samples: {num_data}\n")
                all_results.append(result)
                if num_data >=config["total_sim_num"]:
                    print(f"🎉 Reached the target of {config['total_sim_num']} samples. Stopping further simulations.")
                    break

        except Exception as e:
            print(f"❌ Fail to read or simulate {filename} : {e}")

    # 保存结果
    save_results={"CSI":[], "scatter_positions":[],"Tx_positions":[],"Rx_positions":[],"Tx_orientations":[]}
    for res in all_results:
        save_results["CSI"].extend(res["CSI"])
        save_results["scatter_positions"].extend(res["scatter_positions"])
        save_results["Tx_positions"].extend(res["Tx_positions"])
        save_results["Rx_positions"].extend(res["Rx_positions"])
        save_results["Tx_orientations"].extend(res["Tx_orientations"])
    print(f"\nTotal samples: {len(save_results['CSI'])}")
    # print(f"Sample CSI shape: {np.array(save_results['CSI']).shape}")
    # print(f"Sample scatter_positions shape: {len(save_results['scatter_positions'])}")
    # print(save_results['scatter_positions'])
    np.save(output_path, save_results)
    print(f"\n✅ All completed, save to {output_path}")


if __name__ == "__main__":
    import random
    import tensorflow as tf

    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    tf.random.set_seed(SEED)
    config_folder = "./configs"         
    output_path = "./dataset/sim_results20251017.npy"  # save path
    main(config_folder, output_path)


