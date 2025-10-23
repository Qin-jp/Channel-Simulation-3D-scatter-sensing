import os
os.environ["DRJIT_THREADS"] = "1"
from utils_sionna.ChannelDataGen import gen_CSI_matrix_and_scatter_position_matrix
from utils_sionna.ScatterLocator import locate_scatterers_in_CSI
import numpy as np

import json
import time
import random
import yaml
import re
import drjit as dr
import mitsuba as mi
import multiprocessing as mp
import tempfile
import sys

def run_gen_in_subprocess(config):
    """Run gen_CSI_matrix_and_scatter_position_matrix(config) safely in a new process."""
    import numpy as np
    import os
    import tempfile
    from utils_sionna.ChannelDataGen import gen_CSI_matrix_and_scatter_position_matrix

    # 临时文件路径
    fd, tmp_path = tempfile.mkstemp(suffix=".npy")
    os.close(fd)

    def worker(cfg, path):
        import numpy as np
        from utils_sionna.ChannelDataGen import gen_CSI_matrix_and_scatter_position_matrix
        seed = int(time.time() * 1000) % (2**32 - 1) + os.getpid()
        np.random.seed(seed)
        random.seed(seed)
        tf.random.set_seed(seed)
        try:
            result = gen_CSI_matrix_and_scatter_position_matrix(cfg)
            np.save(path, result, allow_pickle=True)
        except Exception as e:
            np.save(path, {"__error__": str(e)}, allow_pickle=True)
        finally:
            # ✅ 显式释放 Mitsuba / Dr.Jit 资源
            try:
                mi.set_variant(None)                 # 卸载 Mitsuba variant
            except Exception:
                pass
            try:
                import drjit as dr
                dr.sync_thread()                     # 确保线程同步
                dr.flush_malloc_cache()              # 清空内存缓存
            except Exception:
                pass
            sys.exit(0)
    # 启动子进程
    p = mp.Process(target=worker, args=(config, tmp_path))
    p.start()
    p.join(600)  # 最多等待600秒，可按需调大
    if p.is_alive():
        p.terminate()
        p.join()
        raise RuntimeError("Subprocess timed out.")

    # 读取结果
    if not os.path.exists(tmp_path):
        raise RuntimeError("No output file created by subprocess.")
    data = np.load(tmp_path, allow_pickle=True).item()
    os.remove(tmp_path)

    # 检查是否出错
    if isinstance(data, dict) and "__error__" in data:
        raise RuntimeError(f"Subprocess failed: {data['__error__']}")

    return data

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
                #result = gen_CSI_matrix_and_scatter_position_matrix(config)
                try:
                    result = run_gen_in_subprocess(config)
                except Exception as e:
                    print(f"❌ Subprocess failed for {filename}: {e}")
                    continue
                num_data += len(result["CSI"])
                print(f"   Current total samples: {num_data}\n")
                all_results.append(result)
                if num_data >=config["total_sim_num"]:
                    print(f"🎉 Reached the target of {config['total_sim_num']} samples. Stopping further simulations.")
                    break

        except Exception as e:
            print(f"❌ Fail to read or simulate {filename} : {e}")

    # 保存结果
    save_results={"CSI":[],
                 "scatter_positions":[],
                 "Tx_positions":[],
                 "Rx_positions":[],
                 "Tx_orientations":[],
                 "amplitudes":[],
                 "scatterer_indices":[]}
    for res in all_results:
        save_results["CSI"].extend(res["CSI"])
        save_results["scatter_positions"].extend(res["scatter_positions"])
        save_results["Tx_positions"].extend(res["Tx_positions"])
        save_results["Rx_positions"].extend(res["Rx_positions"])
        save_results["Tx_orientations"].extend(res["Tx_orientations"])
        save_results["amplitudes"].extend(res["amp_data"])


    scatterer_indices=locate_scatterers_in_CSI(save_results["scatter_positions"],
                                               save_results["Tx_positions"],
                                               save_results["Rx_positions"],
                                               save_results["Tx_orientations"],
                                               config['Tx_setting']["num_rows"],
                                               config["Tx_setting"]["num_cols"],
                                               config["Rx_setting"]["bandwidth"]/config["Rx_setting"]["num_subcarrier"],
                                               config["Rx_setting"]["num_subcarrier"])
    save_results["scatterer_indices"]=scatterer_indices
    print(f"\nTotal samples: {len(save_results['CSI'])}")
    np.save(output_path, save_results)
    print(f"\n✅ All completed, save to {output_path}")


if __name__ == "__main__":
    import random
    import tensorflow as tf

    # SEED = 42
    # np.random.seed(SEED)
    # random.seed(SEED)
    # tf.random.set_seed(SEED)
    config_folder = "./configs"         
    output_path = "./dataset/sim_results20251023_test.npy"  # save path
    main(config_folder, output_path)


