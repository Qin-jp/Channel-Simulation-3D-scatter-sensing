
import sionna
import numpy as np
import matplotlib.pyplot as plt
#from sionna.rt.utils import OFDMModulator, OFDMDemodulator
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMapSolver, RadioMap
from math import sqrt, log10

import drjit as dr

import matplotlib.pyplot as plt

from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern,\
                      load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, cpx_abs, cpx_convert

def preprocess_complex_csi(sorted_csi):
    # 
    #     将复数 CSI 拆分成实部和虚部并拼接，返回实数张量。
    #     输入 shape: (L, N), dtype: complex64
    #     输出 shape: (L, 2N), dtype: float32
    # 
    real_part = np.real(sorted_csi)
    imag_part = np.imag(sorted_csi)    
        # 沿新维度堆叠 (shape=(L, 2, N))
    csi_realimag = np.stack([real_part, imag_part], axis=1)    
        # 确保 dtype 是 float32
    return csi_realimag.astype(np.float32)
    


def cell_ids_2_positions(cell_ids,rm,radiomap_shape,cell_size):
    """
    将 cell_ids 转换为位置

    参数:
        cell_ids: ndarray，形状为 (N,2)    
        rm: RadioMap 对象
        radiomap_shape: tuple，形状为 (2,)
        cell_size: list，单元格大小,为浮点数二维数组[[0.1,0.1]]  
    返回:
        positions: ndarray，形状为 (N, 3)
    """
    #cell_ids = np.squeeze(cell_ids, axis=0)
    cell_positions = np.zeros((len(cell_ids), 3))
    for i, cell_id in enumerate(cell_ids):
        cell_positions[i,0]=((cell_id[1]-radiomap_shape[1]/2)*cell_size[0,0]+rm._center[0])[0]
        cell_positions[i,1]=((cell_id[0]-radiomap_shape[0]/2)*cell_size[0,0]+rm._center[1])[0]
        cell_positions[i,2]=1.5
    return cell_positions

def position_2_cell_idx(positions,rm,radiomap_shape,cell_size):
    """
    将位置转换为单元格索引
    参数:
        positions: ndarray，形状为 (N,3)
        radiomap_shape: tuple，形状为 (2,)
        cell_size: list，单元格大小,为浮点数二维数组[[0.1,0.1]]   
    返回:
        cell_ids: ndarray，形状为 (N,2)
    """
    cell_ids = np.zeros((len(positions), 2))
    for i, pos in enumerate(positions): 
        cell_ids[i,1]=np.floor((pos[0]-rm._center[0])/cell_size[0,0]+radiomap_shape[1]/2)[0]
        cell_ids[i,0]=np.floor((pos[1]-rm._center[1])/cell_size[0,0]+radiomap_shape[0]/2)[0]
    return cell_ids
# def gen_N_paths(coordstart,coordend,rm,radiomap_shape,cell_size,pathNum,patNum,step):
#     """
#     生成 N 条路径
#     参数:
#         coordstart: list，[x,y]起始坐标
#         coordend: list，[x,y]结束坐标
#         pathNum: int，路径数量
#         patNum: int，路径点数量
#         step: float，步长
#     返回:
#         positions: ndarray，形状为 (N, 3)
#         cell_ids: ndarray，形状为 (N, 2)
#     """
#     coordstart=coordstart+step
#     coordend=coordend-step
#     # print("coordstart: ",coordstart)
#     # print("coordend: ",coordend)
#     positions = np.zeros((pathNum* patNum, 3))
#     cell_ids = np.zeros((pathNum* patNum, 2))
#     for i in range(pathNum):
#         start=[np.random.uniform(coordstart[0],coordend[0]),np.random.uniform(coordstart[1],coordend[1])]
#         theta=np.random.uniform(0,2*np.pi)
#         positionpath=np.zeros((patNum, 3))
#         positionpath[:,0]=np.linspace(start[0],start[0]+np.cos(theta)*(patNum-1)*step,patNum)
#         positionpath[:,1]=np.linspace(start[1],start[1]+np.sin(theta)*(patNum-1)*step,patNum)
#         positionpath[:,2]=np.ones(patNum) * 1.5
#         ceil_ids_path=position_2_cell_idx(positionpath,rm,radiomap_shape,cell_size)
#         positions[i*patNum:(i+1)*patNum]=positionpath
#         cell_ids[i*patNum:(i+1)*patNum]=ceil_ids_path
#     return positions, cell_ids


    

def gen_CSI_matrix_and_scatter_position_matrix(config):
    pathNum = config["pathNum"]  
    velocity_max = config["velocity_max"]
    Tx_setting = config["Tx_setting"]
    Rx_setting = config["Rx_setting"]
    scattering_coefficient = config["scattering_coefficient"]
    scattering_pattern = config["scattering_pattern"]
    cell_size = config["cell_size"]
    bandwidth = config["bandwidth"]
    frequency = config["frequency"]
    
    scene = load_scene(sionna.rt.scene.munich)
    scene.frequency = 30e9
    scene.tx_array = PlanarArray(num_rows=4,
                             num_cols=8,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")

    scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")
    scene.add(Transmitter(name="tx",
                      position=[-100,-150,32],
                      orientation=[0,0,0]))

    rm_solver = RadioMapSolver()

# Configure radio materials for scattering
# By default the scattering coefficient is set to zero
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = 1/sqrt(3) # Try different values in [0,1]
        rm.scattering_pattern = DirectivePattern(alpha_r=10) # Play around with different values of alpha_r

    rm_scat = rm_solver(scene, cell_size=[0.1,0.1], samples_per_tx=int(20e6), max_depth=5,
                    refraction=False, diffuse_reflection=True)

    radiomap_matrix=rm_scat.transmitter_radio_map(metric="path_gain")

    #start_end_cell_ids = [[7500,8100],[8000,8500]]

    #start_end_cell_ids_test = [[7500,8500],[7600,8600]]

    start_end_positions = cell_ids_2_positions(start_end_cell_ids,rm_scat,radiomap_matrix.shape,rm_scat.cell_size)

    # positions,cell_ids=gen_N_paths(start_end_positions[0][0:2],
    #                            start_end_positions[1][0:2],
    #                            rm_scat,
    #                            radiomap_matrix.shape,
    #                            rm_scat.cell_size,
    #                            pathNum=pathNum,patNum=patNum,step=step)
# print("Shape of positions: ", positions.shape)
# print("Shape of cell_ids: ", cell_ids.shape)

    for i,p in enumerate(positions):
        rx = Receiver(name=f"rx-{i}",
                position=p,
                orientation=[0,0,0])
        # 添加用户的随机速度
        scene.add(rx)
    

    bandwidth=100e6 # bandwidth of the receiver (= sampling frequency)
    
#plt.figure()
    p_solver = PathSolver()


# Paths with diffuse reflections
    paths_diff = p_solver(scene,
                 max_depth=5,
                 samples_per_src=10**6,
                 diffuse_reflection=True,
                 refraction=False,
                 synthetic_array=True)


# Compute channel taps wit scattering
    taps_diff = paths_diff.taps(bandwidth, l_min=0, l_max=63, normalize=True, out_type="numpy")
    taps_diff = np.squeeze(taps_diff)
    

# print("Shape of taps_diff: ", taps_diff.shape)

    CSI=np.fft.fft(taps_diff, axis=-1)
    CSI = np.fft.fftshift(CSI, axes=-1)

    return CSI, positions, cell_ids, radiomap_matrix

# print("Shape of CSI: ", CSI.shape)
# print("Shape of Position",positions.shape)



if __name__ == "__main__":
    # CSI_matrix = np.array([])
    # positions_matrix = np.array([])
    # cell_ids_matrix = np.array([])
    Train_MaxTimes=10
    start_end_cell_ids_Train = [[7500,8100],[8000,8500]]
    for i in range(Train_MaxTimes):
        print("第",i,"次")
        # 生成 CSI 矩阵和位置矩阵
        CSI, positions, cell_ids, radiomap = gen_CSI_matrix_and_position_matrix(pathNum=1024,patNum=6,step=0.1,start_end_cell_ids=start_end_cell_ids_Train)
        if i==0:
            CSI_matrix = CSI
            positions_matrix = positions
            cell_ids_matrix = cell_ids
        else:
            CSI_matrix=np.concatenate((CSI_matrix, CSI), axis=0)
            positions_matrix=np.concatenate((positions_matrix, positions), axis=0)
            cell_ids_matrix=np.concatenate((cell_ids_matrix, cell_ids), axis=0)
    # 先做预处理，把复数拆成实+虚
    CSI_realimag = preprocess_complex_csi(CSI_matrix)

    #print("Shape of CSI_realimag: ", CSI_realimag.shape)

    CSI_realimag=np.reshape(CSI_realimag, (CSI_realimag.shape[0]//6,6,2,32,64))
    positions_matrix=np.reshape(positions_matrix, (positions_matrix.shape[0]//6,6,3))
    cell_ids_matrix=np.reshape(cell_ids_matrix, (cell_ids_matrix.shape[0]//6,6,2))
    # 保存多个矩阵（.npz）
    np.savez("train.npz", CSI=CSI_realimag[0:9216], Pos=positions_matrix[0:9216], CellIdx=cell_ids_matrix[0:9216], radiomap=radiomap)
    np.savez("val.npz", CSI=CSI_realimag[9216:], Pos=positions_matrix[9216:], CellIdx=cell_ids_matrix[9216:], radiomap=radiomap)

    Test_MaxTimes=1
    #start_end_cell_ids_Test = [[7500,8500],[7600,8600]]
    start_end_cell_ids_Test = [[7500,8100],[8000,8500]]
    for i in range(Test_MaxTimes):
        print("第",i,"次")
        # 生成 CSI 矩阵和位置矩阵
        CSI, positions, cell_ids, radiomap = gen_CSI_matrix_and_position_matrix(pathNum=1024,patNum=6,step=0.1,start_end_cell_ids=start_end_cell_ids_Test)
        if i==0:
            CSI_matrix = CSI
            positions_matrix = positions
            cell_ids_matrix = cell_ids
        else:
            CSI_matrix=np.concatenate((CSI_matrix, CSI), axis=0)
            positions_matrix=np.concatenate((positions_matrix, positions), axis=0)
            cell_ids_matrix=np.concatenate((cell_ids_matrix, cell_ids), axis=0)
    # 先做预处理，把复数拆成实+虚
    CSI_realimag = preprocess_complex_csi(CSI_matrix)

    #print("Shape of CSI_realimag: ", CSI_realimag.shape)

    CSI_realimag=np.reshape(CSI_realimag, (CSI_realimag.shape[0]//6,6,2,32,64))
    positions_matrix=np.reshape(positions_matrix, (positions_matrix.shape[0]//6,6,3))
    cell_ids_matrix=np.reshape(cell_ids_matrix, (cell_ids_matrix.shape[0]//6,6,2))
    # 保存多个矩阵（.npz）
    np.savez("test.npz", CSI=CSI_realimag, Pos=positions_matrix, CellIdx=cell_ids_matrix, radiomap=radiomap)