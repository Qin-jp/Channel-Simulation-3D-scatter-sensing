import numpy as np
import ScatterLocator
import json
import torch

def serialize_sparse_tensor(t):
    return {
        "indices": t.indices(),
        "values": t.values(),
        "size": t.size()
    }

def deserialize_sparse_tensor(d):
    return torch.sparse_coo_tensor(d["indices"], d["values"], d["size"])

def aggregate_scatterer_bias(scatterer_indices,num_row,num_col,left_num_delay,scaling_ratio):
    num_row_output = int(num_row * scaling_ratio)
    num_col_output = int(num_col * scaling_ratio)
    left_num_delay_output = int(left_num_delay * scaling_ratio)

    clean_scatterer_indices=[]
    for idx in range(len(scatterer_indices)):
        if scatterer_indices[idx][2]>=left_num_delay_output + 1:
            continue
        clean_scatterer_indices.append(scatterer_indices[idx])
    scatterer_indices=np.array(clean_scatterer_indices)

    point_bias_list=np.empty((num_row_output,num_col_output,left_num_delay_output),dtype=object)
    for i in range(num_row_output):
        for j in range(num_col_output):
            for k in range(left_num_delay_output):
                point_bias_list[i, j, k] = []
    for idx in range(len(scatterer_indices)):
        x,y,z=scatterer_indices[idx]*scaling_ratio
        xi = int(np.clip(np.floor(x), 0, num_row_output - 1))
        yi = int(np.clip(np.floor(y), 0, num_col_output - 1))
        zi = int(np.clip(np.floor(z), 1, left_num_delay_output)) - 1
        point_bias_list[xi,yi,zi].append(np.array([x-np.round(x),y-np.round(y),z-np.round(z)]))

    output=np.zeros((num_row_output,num_col_output,left_num_delay_output,3+1))
    indices = []
    values = []
    seen = set()
    for idx in range(len(scatterer_indices)):
        x,y,z=scatterer_indices[idx]*scaling_ratio
        xi = int(np.clip(np.floor(x), 0, num_row_output - 1))
        yi = int(np.clip(np.floor(y), 0, num_col_output - 1))
        zi = int(np.clip(np.floor(z), 1, left_num_delay_output)) - 1
        if zi == -1:
            continue
        if output[xi,yi,zi,3]==1.0:
            continue
        if len(point_bias_list[xi,yi,zi])==1:
            output[xi,yi,zi,:3]=point_bias_list[xi,yi,zi][0]
            output[xi,yi,zi,3]=1.0
        elif len(point_bias_list[xi,yi,zi]) > 1:
            bias_sum=np.zeros((3,))
            for cluster_id in range(len(point_bias_list[xi,yi,zi])):
                bias_sum+=point_bias_list[xi,yi,zi][cluster_id]
            output[xi,yi,zi,:3]=bias_sum/len(point_bias_list[xi,yi,zi])
            output[xi,yi,zi,3]=1
        key = (xi, yi, zi)
        if key in seen:
            continue
        seen.add(key)
        indices.append([xi, yi, zi])
        values.append(np.concatenate([output[xi,yi,zi,:3], [1.0]]))
    indices = torch.tensor(np.array(indices).T, dtype=torch.long)  # (3, N)
    values = torch.tensor(np.array(values), dtype=torch.float32)   # (N, 4)

    if len(indices) == 0:
        return torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long),
            torch.empty((0, 4), dtype=torch.float32),
            size=(num_row_output, num_col_output, left_num_delay_output, 4)
        )

    # 构造稀疏张量
    sparse_output = torch.sparse_coo_tensor(
        indices,
        values,
        size=(num_row_output, num_col_output, left_num_delay_output, 4)
    ).coalesce()

    return sparse_output

if __name__=="__main__":


    data=np.load("/home/jingpeng/graduation_project/Channel_Simulation/dataset/sim_results20251023.npy", allow_pickle=True)
    print(data.item().keys())
    CSI=data.item().get("CSI")
    scatter_positions=data.item().get("scatter_positions")
    Tx_positions=data.item().get("Tx_positions")
    Rx_positions=data.item().get("Rx_positions")
    Tx_orientations=data.item().get("Tx_orientations")
    indices=ScatterLocator.locate_scatterers_in_CSI(scatter_positions,Tx_positions,Rx_positions,Tx_orientations,64,64,100e6/1024,1024)
    #print(indices)

    with open("/home/jingpeng/graduation_project/Channel_Simulation/configs/sim_config1.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    save_res={"GF":[[],[],[]],"CSI":[]}

    for idx in range(len(indices)):
        index=indices[idx]
        save_groundtruth0=aggregate_scatterer_bias(index,config["Tx_setting"]["num_rows"],config["Tx_setting"]["num_cols"],64,0.25)
        save_groundtruth1=aggregate_scatterer_bias(index,config["Tx_setting"]["num_rows"],config["Tx_setting"]["num_cols"],64,0.5)
        save_groundtruth2=aggregate_scatterer_bias(index,config["Tx_setting"]["num_rows"],config["Tx_setting"]["num_cols"],64,1)
        if save_groundtruth0._nnz()==0 and save_groundtruth1._nnz()==0 and save_groundtruth2._nnz()==0:
            continue
        save_CSI=CSI[idx][:,:,:,1:65]
        save_res["GF"][0].append(save_groundtruth0)
        save_res["GF"][1].append(save_groundtruth1)
        save_res["GF"][2].append(save_groundtruth2)
        save_res["CSI"].append(save_CSI)

    save_data = {
        "GF": [
            [serialize_sparse_tensor(t) for t in save_res["GF"][0]],
            [serialize_sparse_tensor(t) for t in save_res["GF"][1]],
            [serialize_sparse_tensor(t) for t in save_res["GF"][2]],
        ],
        "CSI": save_res["CSI"]
    }

    torch.save(save_data, "/home/jingpeng/graduation_project/Channel_Simulation/dataset/save_res20251024.pt")    


