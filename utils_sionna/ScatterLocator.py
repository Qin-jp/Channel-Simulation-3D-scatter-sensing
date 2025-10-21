import numpy as np

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
    n, num_subcarriers, __  = CSI.shape
    #assert num_ant == num_rows * num_cols, "Antenna number mismatch!"

    CSI_angle_delay = np.zeros((n, num_subcarriers, num_rows, num_cols), dtype=np.complex64)

    for i in range(n):
        # reshape antenna dimension into 2D array (rows x cols)
        H_space_freq = CSI[i].reshape(num_subcarriers, num_rows, num_cols)

        # 2D FFT over antenna array (spatial → angular domain)
        H_angle_freq = np.fft.fft2(H_space_freq, axes=(1, 2))

        # 1D IFFT over frequency domain (frequency → delay domain)
        H_angle_delay = np.fft.ifft(H_angle_freq, axis=0)

        CSI_angle_delay[i] = H_angle_delay

    return CSI_angle_delay

def compute_AoD(tx_pos, scatter_pos, tx_angle):
    """Compute the Angle of Departure (AoD) from the transmitter to the scatterer.
    Args:
        tx_pos (np.ndarray): The position of the transmitter (3,).
        scatter_pos (np.ndarray): The position of the scatterer (3,).
        tx_angle (np.ndarray): The orientation of the transmitter in Sionna form (3,).
    Returns:
        tuple: A tuple containing the elevation angle (theta) and azimuth angle (phi) in radians.
    """
    tx_theta=tx_angle[1]
    tx_phi=tx_angle[0]
    v=scatter_pos-tx_pos
    v_theta=np.arccos(v[2]/np.linalg.norm(v))#?????
    v_phi=np.arctan2(v[1],v[0])#?????
    theta=v_theta-tx_theta
    phi=v_phi-tx_phi    
    return theta, phi

def compute_CSI_from_one_path(Tx_ant_rows,
                              Tx_ant_cols,
                              num_subcarriers,
                              carrier_frequency,
                              bandwidth,
                              amplitude,
                              tau):                              
    f_0=carrier_frequency - bandwidth/2
    delta_f=bandwidth/num_subcarriers
    delay_vector=[np.exp(-1j*2*np.pi*(f_0+nc*delta_f)*tau) for nc in range(num_subcarriers)]
    amplitude=(np.reshape(amplitude,(Tx_ant_rows,Tx_ant_cols)))
    h=[delay_vector[i]*amplitude for i in range(len(delay_vector))]
    return np.reshape(h,(num_subcarriers, Tx_ant_rows*Tx_ant_cols))


def reconstruct_CSI_from_scatterers_and_amplitudes(Tx_position,
                                                   scatter_positions,
                                                   amplitudes,
                                                   Tx_orientation,
                                                   num_subcarriers,
                                                   Tx_ant_rows,
                                                   Tx_ant_cols,
                                                   carrier_frequency,
                                                   bandwidth,
                                                   taus,
                                                   speed_of_light=3e8):

    """Reconstruct the CSI matrix from scatterer positions and their amplitudes.

    Args:
        Tx_position (np.ndarray): The position of the transmitter (3,).
        scatter_positions (np.ndarray): The scatterer positions (num_scatterers, 3).
        amplitudes (np.ndarray): The complex amplitudes of the scatterers (num_scatterers,).
        Tx_orientation (np.ndarray): The orientation of the transmitter in Sionna form (3,).
        num_subcarriers (int): The number of subcarriers.
        Tx_ant_rows (int): The number of transmitter antenna rows.
        Tx_ant_cols (int): The number of transmitter antenna columns.
        carrier_frequency (float): The carrier frequency in Hz.
        bandwidth (float): The bandwidth in Hz.
        speed_of_light (float): The speed of light in m/s. Default is 3e8 m/s.

    Returns:
        np.ndarray: The reconstructed CSI matrix of shape (2, num_subcarriers, num_rows, num_cols, ).
    """
    #print(scatter_positions.shape)
    #print(amplitudes.shape)
    h=np.zeros((num_subcarriers, Tx_ant_rows*Tx_ant_cols), dtype=np.complex64)
    for scatter_id in range(len(scatter_positions)):
        #scatter_pos = scatter_positions[scatter_id]
        amplitude = amplitudes[scatter_id,0]
        tau=taus[scatter_id]
        #tx_angle = Tx_orientation
        #theta, phi = compute_AoD(np.squeeze(Tx_position),np.squeeze(scatter_pos), np.squeeze(tx_angle))
        h+=compute_CSI_from_one_path(Tx_ant_rows,
                                     Tx_ant_cols,
                                     num_subcarriers,
                                     carrier_frequency,
                                     bandwidth,
                                     amplitude,
                                     tau)

    CSI=np.expand_dims(h,axis=0)
    CSI = AntennaFreq2AngleDelay(CSI, num_rows=Tx_ant_rows, num_cols=Tx_ant_cols)
    return CSI


def light_spot_index2theta_phi_tau(index,num_row,num_col,delta_f,num_subcarrier):
    row_index=index[0]
    col_index=index[1]
    tau_index=index[2]
    if col_index<num_col/2:
        cos_theta=col_index/(num_col/2)
        theta=np.arccos(cos_theta)
    else:
        cos_theta=(col_index-num_col)/(num_col/2)
        theta=np.arccos(cos_theta)

    if row_index<num_row/2:
        sin_phi=row_index/(num_row/2)/np.sin(theta)
        phi=np.arcsin(sin_phi)
    else:
        sin_phi=(row_index-num_row)/(num_row/2)/np.sin(theta)
        phi=np.arcsin(sin_phi)
    tau=tau_index/(delta_f*num_subcarrier)
    return theta,phi,tau

def theta_phi_tau2light_spot_index(theta,phi,tau,num_row,num_col,delta_f,num_subcarrier):
    index=np.zeros((3,1))
    if np.cos(theta)<0:
        index[1]=(1+np.cos(theta)/2)*num_col
    else:
        index[1]=np.cos(theta)/2*num_col
    if np.sin(phi)<0:
        index[0]=(1+np.sin(theta)*np.sin(phi)/2)*num_row
    else:
        index[0]=np.sin(theta)*np.sin(phi)/2*num_row
    index[2]=delta_f*tau*num_subcarrier
    return index

def theta_phi_tau2scatterer_pos(Tx_pos,Rx_pos,theta,phi,tau):
    Tx_pos=np.squeeze(Tx_pos)
    Rx_pos=np.squeeze(Rx_pos)
    lightspeed=3e8
    c=np.linalg.norm(Tx_pos-Rx_pos)/2
    a=(tau*lightspeed+2*c)/2
    scatter_direction_vecter=np.array([np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta)])
    Rx_Tx_direction_vecter=(Rx_pos-Tx_pos)/2/c
    cos_alpha=np.dot(scatter_direction_vecter,Rx_Tx_direction_vecter)
    e=c/a
    length=(a*(1-e*e))/(1-e*cos_alpha)
    return Tx_pos+length*scatter_direction_vecter

def locate_scatterers_in_CSI(CSI, scatter_positions, Tx_positions, Rx_positions, Tx_orientations):
    """Locate the scatterers in the 3D CSI matrix.

    Args:
        CSI (np.ndarray): The CSI matrix of shape (num_samples, 2, Tx_ant_row, Tx_ant_col, num_subcarriers).
        scatter_positions (np.ndarray): The scatterer positions of shape (num_samples, num_scatterers, 3).
        Tx_positions (np.ndarray): The transmitter positions of shape (num_samples, 3).
        Rx_positions (np.ndarray): The receiver positions of shape (num_samples, 3).
        scene (object): The scene object containing the environment information.
    Returns:
        dict: A dictionary containing the located scatterers and their corresponding paths.
    """
    for id in range(len(CSI)):
        #csi_sample = CSI[id]
        scatter_pos_sample = scatter_positions[id]
        tx_pos = Tx_positions[id]
        rx_pos = Rx_positions[id]
        num_scatterers = scatter_pos_sample.shape[0]

        # Compute the distances and angles for each scatterer
        for i in range(num_scatterers):
            scatter_pos = scatter_pos_sample[i]
            # d_tx_scatter = np.linalg.norm(scatter_pos - tx_pos)
            # d_scatter_rx = np.linalg.norm(rx_pos - scatter_pos)
            # total_distance = d_tx_scatter + d_scatter_rx

            # Compute angles
            tx_angle = Tx_orientations[id] # Assuming scene has a method to get Tx orientation
            theta, phi = compute_AoD(np.squeeze(tx_pos),np.squeeze(scatter_pos), np.squeeze(tx_angle))
            print("theta:",theta,"phi:",phi)

            # Here you would map the distance and angles to the CSI matrix indices
            # This part is highly dependent on how your CSI matrix is structured
            # and how distances/angles correspond to indices.

            # For demonstration, let's assume we have a function that does this mapping:
            # row_idx, col_idx, subcarrier_idx = map_to_csi_indices(total_distance, theta, phi, csi_sample.shape)

            # Now you can locate the scatterer in the CSI matrix
            # amplitude = np.abs(csi_sample[0, row_idx, col_idx, subcarrier_idx])
            # phase = np.angle(csi_sample[1, row_idx, col_idx, subcarrier_idx])

            # Store or print the located scatterer information
            # print(f"Scatterer {i}: Distance={total_distance}, Theta={theta}, Phi={phi}, Amplitude={amplitude}, Phase={phase}")



if __name__ == "__main__":
    data=np.load("/home/jingpeng/graduation_project/Channel_Simulation/dataset/sim_results20251020.npy", allow_pickle=True)
    print(data.item().keys())
    CSI=data.item().get("CSI")
    scatter_positions=data.item().get("scatter_positions")
    Tx_positions=data.item().get("Tx_positions")
    Rx_positions=data.item().get("Rx_positions")
    Tx_orientations=data.item().get("Tx_orientations")
    amplitude=data.item().get("amplitudes")

    taus=[np.zeros(len(scatter_positions[i])) for i in range(len(scatter_positions))]
    for i in range(len(Tx_positions)):
        for j in range(len(scatter_positions[i])):
            taus[i][j]=(np.linalg.norm(np.reshape(Tx_positions[i],(3,)) - scatter_positions[i][j])
                        + np.linalg.norm(np.reshape(Rx_positions[i],(3,)) - scatter_positions[i][j])
                        - np.linalg.norm(Tx_positions[i] - Rx_positions[i])) / 3e8

    TestCSI=[]
    for i in range(8):
        TestCSI.append(reconstruct_CSI_from_scatterers_and_amplitudes(Tx_positions[i],
                                                    scatter_positions[i],
                                                    amplitude[i],
                                                    Tx_orientations[i],
                                                    num_subcarriers=1024,
                                                    Tx_ant_rows=16,
                                                    Tx_ant_cols=16,
                                                    carrier_frequency=3e9,
                                                    bandwidth=100e6,
                                                    speed_of_light=3e8,
                                                    taus=taus[i]))
    save_results={"CSI":TestCSI}
    #np.save("/home/jingpeng/graduation_project/Channel_Simulation/debug_file/test_reconstructed_CSI.npy", save_results)
    #locate_scatterers_in_CSI(CSI, scatter_positions, Tx_positions, Rx_positions, Tx_orientations)