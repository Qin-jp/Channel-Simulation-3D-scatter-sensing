import sionna
import numpy as np
from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, RadioMapSolver, RadioMap
from sionna.rt import LambertianPattern, DirectivePattern, BackscatteringPattern,\
                      load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                      PathSolver, RadioMapSolver, cpx_abs, cpx_convert

import mitsuba as mi

def preprocess_complex_csi(sorted_csi):
    """
    Preprocess complex CSI by separating real and imaginary parts.
    Args:
        sorted_csi: ndarray, shape (L, num_rows, num_cols, num_subcarrier), dtype: complex64
    Returns:
        csi_realimag: ndarray, shape (2, L, num_rows, num_cols, num_subcarrier), dtype: float32
    """
    real_part = np.real(sorted_csi)
    imag_part = np.imag(sorted_csi)    
    csi_realimag = np.stack((real_part, imag_part), axis=0)
    return csi_realimag.astype(np.float32)
    
def cell_ids_2_positions(cell_ids,rm,radiomap_shape,cell_size):
    """
    Convert cell_ids to positions.

    Args:
        cell_ids: ndarray, shape (N, 2)  
        rm: RadioMap object  
        radiomap_shape: tuple, shape (2,)  
        cell_size: list, the size of each cell, a 2D float array like [[0.1, 0.1]]  

    Returns:
        positions: ndarray, shape (N, 3)
    """
    cell_positions = np.zeros((len(cell_ids), 3))
    for i, cell_id in enumerate(cell_ids):
        cell_positions[i,0]=((cell_id[1]-radiomap_shape[1]/2)*cell_size[0,0]+rm._center[0])[0]
        cell_positions[i,1]=((cell_id[0]-radiomap_shape[0]/2)*cell_size[0,0]+rm._center[1])[0]
        cell_positions[i,2]=1.5
    return cell_positions

def position_2_cell_idx(positions,rm,radiomap_shape,cell_size):
    """
    Convert positions to cell indices.

    Args:
        positions: ndarray, shape (N, 3)  
        radiomap_shape: tuple, shape (2,)  
        cell_size: list, the size of each cell, a 2D float array like [[0.1, 0.1]]  

    Returns:
        cell_ids: ndarray, shape (N, 2)
    """
    cell_ids = np.zeros((len(positions), 2))
    for i, pos in enumerate(positions): 
        cell_ids[i,1]=np.floor((pos[0]-rm._center[0])/cell_size[0,0]+radiomap_shape[1]/2)[0]
        cell_ids[i,0]=np.floor((pos[1]-rm._center[1])/cell_size[0,0]+radiomap_shape[0]/2)[0]
    return cell_ids

def create_scene(config):
    Tx_setting = config["Tx_setting"]
    Rx_setting = config["Rx_setting"]
    scattering_coefficient = config["scattering_coefficient"]
    scattering_pattern_alpha = config["scattering_pattern_alpha"]
    frequency = config["frequency"]
    
    scene = load_scene(sionna.rt.scene.munich)
    scene.frequency = frequency
    scene.tx_array = PlanarArray(num_rows=Tx_setting["num_rows"],
                                 num_cols=Tx_setting["num_cols"],
                                 vertical_spacing=Tx_setting["vertical_spacing"],
                                 horizontal_spacing=Tx_setting["horizontal_spacing"],
                                 pattern=Tx_setting["antenna pattern"],#"tr38901"
                                 polarization=Tx_setting["polarization"])#"V"

    scene.rx_array = PlanarArray(num_rows=Rx_setting["num_rows"],
                                 num_cols=Rx_setting["num_cols"],
                                 vertical_spacing=Rx_setting["vertical_spacing"],
                                 horizontal_spacing=Rx_setting["horizontal_spacing"],
                                 pattern=Rx_setting["antenna pattern"],#"tr38901"
                                 polarization=Rx_setting["polarization"])#"V
    
    scene.add(Transmitter(name="Tx",
              position=Tx_setting["position"],#[-100,-150,32]
              orientation=Tx_setting["orientation"]))#[0,0,0]
    # Configure radio materials for scattering because by default the scattering coefficient is set to zero
    for rm in scene.radio_materials.values():
        rm.scattering_coefficient = scattering_coefficient # 1/sqrt(3)
        rm.scattering_pattern = DirectivePattern(alpha_r=scattering_pattern_alpha) # 10

    return scene

def sample_rx(Tx_position, Rx_height_max, sample_radis, num_samples):
    """
    Sample receiver positions around a transmitter position.

    Args:
        Tx_position: list, shape (3,)  
        Rx_height_max: float, maximum height of the receiver  
        sample_radis: float, sampling radius
        num_samples: int, number of samples  

    Returns:
        positions: ndarray, shape (num_samples, 3)
    """
    Tx_position = np.array(Tx_position).squeeze()
    positions = np.zeros((num_samples, 3))
    for i in range(num_samples):
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        distance = np.random.uniform(0, sample_radis)
        positions[i, 0] = Tx_position[0] + distance * np.cos(angle)
        positions[i, 1] = Tx_position[1] + distance * np.sin(angle)
        positions[i, 2] = np.random.uniform(1, Rx_height_max)
    return positions

def filter_zero_CSI(CSI,Rx_positions):
    """
    Filter out zero CSI entries.

    Args:
        CSI: ndarray, shape (L, N), dtype: complex64  

    Returns:
        filtered_CSI: ndarray, shape (L, M), dtype: complex64  
        valid_indices: list, indices of non-zero CSI entries
    """
    valid_indices = [i for i in range(CSI.shape[0]) if np.sum(np.abs(CSI[i,:,:])) > 0.1]
    filtered_CSI = CSI[valid_indices,:,:]
    filtered_positions = Rx_positions[valid_indices,:]
    return filtered_CSI, filtered_positions

def AntennaFreq2AngleDelay(CSI, num_rows=4, num_cols=4):
    """
    Convert CSI from antenna-frequency domain to angle-delay domain.

    Args:
        CSI: np.ndarray, shape (n, 16, 64), complex64
        num_rows: number of antenna rows (default 4)
        num_cols: number of antenna columns (default 4)

    Returns:
        CSI_angle_delay: np.ndarray, shape (n, num_rows, num_cols, 64), complex64
    """
    #print(CSI.shape)
    n, num_ant, num_subcarriers = CSI.shape
    assert num_ant == num_rows * num_cols, "Antenna number mismatch!"

    CSI_angle_delay = np.zeros((n, num_rows, num_cols, num_subcarriers), dtype=np.complex64)

    for i in range(n):
        # reshape antenna dimension into 2D array (rows x cols)
        H_space_freq = CSI[i].reshape(num_rows, num_cols, num_subcarriers)

        # 2D FFT over antenna array (spatial → angular domain)
        H_angle_freq = np.fft.fft2(H_space_freq, axes=(0, 1))

        # 1D IFFT over frequency domain (frequency → delay domain)
        H_angle_delay = np.fft.ifft(H_angle_freq, axis=-1)

        CSI_angle_delay[i] = H_angle_delay

    return CSI_angle_delay

def get_scatter_pos_and_attached_data(paths_obj,CSI,scene,save_path_without_LoS=False):
    """
    Extract scatterer positions and associated data from paths object and CSI.

    Args:
        paths_obj: PathSolver object containing path information
        CSI: ndarray
    """
    a=np.array(paths_obj.a)
    complex_a=a[0,...]+1j*a[1,...]
    interactions=np.array(paths_obj.interactions)
    vertices=np.array(paths_obj.vertices)
    Tx_data=[]
    Rx_data=[]
    CSI_data=[]
    Scatter_pos_data=[]
    amp_data=[]
    Tx_id=0
    for Rx_id in range(interactions.shape[1]):
        max_mag = np.max(np.sum(np.abs(complex_a[Rx_id,0,Tx_id,:,:]),axis=-2))
        mag_threshold = max_mag * 0.1  # 5% of the maximum magnitude
        significant_paths = np.sum(np.abs(complex_a[Rx_id,0,Tx_id,:,:]),axis=-2)> mag_threshold    
        reflection_paths=[x or y for x,y in zip(interactions[0,Rx_id,Tx_id,:]==1, interactions[0,Rx_id,Tx_id,:]==2)]
        
        LoS_path = [x and y for x,y in zip(interactions[0,Rx_id,Tx_id,:]==0 , significant_paths)]
        if np.sum(LoS_path)>0:
            #Scatter_pos_data.append(np.array(scene.receivers[f"Rx-{Rx_id}"].position).reshape(1,3))
            #amp_data.append(complex_a[Rx_id,:,Tx_id,:,LoS_path])
            valid_paths = significant_paths & reflection_paths
            if np.sum(valid_paths)==0:
                continue
            Scatter_pos = vertices[0,Rx_id,Tx_id,valid_paths,:] #(N, 3)
            Tx_data.append(np.array(scene.transmitters["Tx"].position))
            Rx_data.append(np.array(scene.receivers[f"Rx-{Rx_id}"].position))
            CSI_data.append(CSI[:,Rx_id,...])
            Scatter_pos_data.append(Scatter_pos)
            amp_data.append(complex_a[Rx_id,:,Tx_id,:,valid_paths])
            #Scatter_pos_data[len(Scatter_pos_data)-1]=np.append(Scatter_pos_data[len(Scatter_pos_data)-1],Scatter_pos,axis=0)
            #amp_data[len(amp_data)-1]=np.append(amp_data[len(amp_data)-1],complex_a[Rx_id,:,Tx_id,:,valid_paths],axis=0)
        else:
            if save_path_without_LoS == False:
                continue
            valid_paths = significant_paths & reflection_paths
            if np.sum(valid_paths)==0:
                continue
            Scatter_pos = vertices[0,Rx_id,Tx_id,valid_paths,:] #(N, 3)
            Tx_data.append(np.array(scene.transmitters["Tx"].position))
            Rx_data.append(np.array(scene.receivers[f"Rx-{Rx_id}"].position))
            CSI_data.append(CSI[:,Rx_id,...])
            Scatter_pos_data.append(Scatter_pos)
            amp_data.append(complex_a[Rx_id,:,Tx_id,:,valid_paths])
        print("Rx-",Rx_id," num valid scatterers:",np.sum(valid_paths))
    return CSI_data,Scatter_pos_data,Tx_data,Rx_data,amp_data

def gen_CSI_matrix_and_scatter_position_matrix(config):
    """
    Generate CSI matrix and scatter position matrix based on the given configuration.
    Args:
        config: dict, configuration parameters
    Returns:
        dict: containing "CSI" and "scatter positions" and other related data
    """
    velocity_max = config["velocity_max"]
    Rx_setting = config["Rx_setting"]
    Tx_setting = config["Tx_setting"]
    cell_size = config["cell_size"]
    
    scene = create_scene(config)

    positions = sample_rx(scene.transmitters["Tx"].position, config["Rx_height_max"], config["sample_radis"], num_samples=config["num_samples"])

    for i,p in enumerate(positions):
        speed = np.random.uniform(0, velocity_max) 
        theta = np.random.uniform(0, np.pi) if p[2] > 2 else 0.5 * np.pi #consider horizontal movement if height > 2m
        phi = np.random.uniform(0, 2 * np.pi)
        velocity = mi.Vector3f(
        [float(speed * np.sin(theta) * np.cos(phi)),
        float(speed * np.sin(theta) * np.sin(phi)),
        float(speed * np.cos(theta))]
        ) if speed > 0 else mi.Vector3f(0,0,0) #the Vecter3f() function requires float input, not np.float32
        rx = Receiver(name=f"Rx-{i}",
                position=p,
                orientation=Rx_setting["orientation"],#[0,0,0]
                velocity=velocity)        
        scene.add(rx)
    

    bandwidth=Rx_setting["bandwidth"] # bandwidth of the receiver (= sampling frequency)
    
    #plt.figure()
    #scene.preview()
    
    p_solver = PathSolver()


# Paths with diffuse reflections
    paths_diff = p_solver(scene,
                 max_depth=1,
                 samples_per_src=10**4,
                 diffuse_reflection=True,
                 refraction=False,
                 synthetic_array=True)   

# Compute channel taps with scattering
    frequencies= np.linspace(scene.frequency[0] - bandwidth / 2, scene.frequency[0] + bandwidth / 2, Rx_setting["num_subcarrier"])
    
    h_freq=paths_diff.cfr(frequencies=frequencies, normalize=True, out_type="numpy")
    print("h_freq shape:", h_freq.shape)
    h_freq =np.reshape(h_freq,(h_freq.shape[0],Tx_setting["num_rows"]*Tx_setting["num_cols"],h_freq.shape[-1])) #(Rx_num, num_row, num_col num_subcarrier)

    # CSI = np.fft.fft(taps_diff, axis=-1)
    CSI = h_freq
    CSI = AntennaFreq2AngleDelay(CSI, num_rows=Tx_setting["num_rows"], num_cols=Tx_setting["num_cols"])
    CSI = preprocess_complex_csi(CSI)
    print("CSI shape:", CSI.shape)

    #extract valid scatter positions and attached data

    CSI,scatter_pos,Tx_pos,Rx_pos,amp_data=get_scatter_pos_and_attached_data(paths_diff,CSI,scene)


    return {"CSI":CSI,
            "scatter_positions":scatter_pos,
            "Tx_positions":Tx_pos,
            "Rx_positions":Rx_pos,
            "amp_data":amp_data,
            "Tx_orientations":np.array(np.repeat(scene.transmitters["Tx"].orientation,len(CSI),axis=-1)).reshape(len(CSI),3)}

