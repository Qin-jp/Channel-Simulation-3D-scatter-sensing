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
    v_theta=np.arccos(v[2]/np.linalg.norm(v))
    v_phi=np.arctan2(v[1],v[0])
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
    h=np.zeros((num_subcarriers, Tx_ant_rows*Tx_ant_cols), dtype=np.complex64)
    for scatter_id in range(len(scatter_positions)):
        amplitude = amplitudes[scatter_id,0]
        tau=taus[scatter_id]
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
    """Convert a 3D light spot index into angular and delay domain parameters.

    This function maps a discrete 3D index (row, column, delay) from a light spot 
    or antenna array representation into its corresponding physical parameters:
    elevation angle (theta), azimuth angle (phi), and path delay (tau).

    Args:
        index (tuple or list of int): A 3-element index (row_index, col_index, tau_index)
            representing the position in the angular-delay grid.
        num_row (int): The number of rows in the grid (corresponding to the azimuth dimension).
        num_col (int): The number of columns in the grid (corresponding to the elevation dimension).
        delta_f (float): The subcarrier spacing in Hz.
        num_subcarrier (int): The total number of subcarriers used in the frequency domain.

    Returns:
        tuple:
            - theta (float): Polar angle in radians, derived from the column index.
            - phi (float): Azimuth angle in radians, derived from the row index.
            - tau (float): Propagation delay in seconds, derived from the delay index.
    """
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
    """Map the incident path parameters (theta, phi, tau) to their corresponding indices 
    in the CSI light spot representation.

    Args:
        theta (float): The polar angle in the spherical coordinate system, measured from 
            the +z axis downward, in the range [0, π].  
            (Note: If using elevation angle, convert via θ = π/2 - elevation.)
        phi (float): The azimuth angle in the xy-plane, measured counterclockwise 
            from the +x axis toward the +y axis, in the range [-π, π].
        tau (float): Path propagation delay in seconds, determining the frequency-domain shift in CSI.
        num_row (int): Number of rows in the CSI (vertical spatial dimension of the array).
        num_col (int): Number of columns in the CSI (horizontal spatial dimension of the array).
        delta_f (float): Subcarrier spacing in Hz.
        num_subcarrier (int): Number of subcarriers used for normalization in the delay dimension.

    Returns:
        np.ndarray: A 3-element array representing the [row_index, col_index, depth_index], where:
            - index[0]: Vertical index, derived from sin(theta) * sin(phi)
            - index[1]: Horizontal index, derived from cos(theta)
            - index[2]: Frequency/delay index, derived from tau

    Notes:
        - The horizontal (column) mapping uses cos(theta) to distinguish directions above or below the z-axis.
        - The vertical (row) mapping uses sin(theta)*sin(phi) to represent elevation and azimuth jointly.
        - The delay index is computed as delta_f * tau * num_subcarrier, corresponding to its 
          relative location in the frequency-domain CSI representation.
    """
    index=np.zeros((3,))
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
    """Compute the 3D position of a scatterer based on geometric parameters derived 
    from CSI, assuming the scatterer lies on an ellipsoid formed by constant path delay.

    Args:
        Tx_pos (np.ndarray): The 3D position of the transmitter, shape (3,).
        Rx_pos (np.ndarray): The 3D position of the receiver, shape (3,).
        theta (float): Polar angle (measured from +z axis) in spherical coordinates, in radians.
        phi (float): Azimuth angle (measured counterclockwise from +x axis in the xy-plane), in radians.
        tau (float): Path delay in seconds, representing the total propagation time from Tx to Rx via the scatterer.

    Returns:
        np.ndarray: The estimated 3D position of the scatterer, shape (3,).

    Notes:
        - The locus of all possible scatterer positions corresponding to a given delay τ 
          is an ellipsoid with the Tx and Rx as its two foci.
        - The semi-major axis `a` of the ellipsoid is determined from the total path length
          `a = (c * τ + 2 * c0) / 2`, where `c0` is half the Tx–Rx distance.
        - The eccentricity `e = c0 / a` characterizes how elongated the ellipsoid is.
        - The intersection point between the ellipsoid and the ray defined by (θ, φ)
          from the transmitter gives the scatterer position.

    """
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

def locate_scatterers_in_real_world(light_spot_indices, Tx_positions, Rx_positions, Tx_orientations, num_row,num_col,delta_f,num_subcarrier):
    """Estimate the 3D positions of scatterers from CSI light spot indices.

    Args:
        light_spot_indices (list[np.ndarray]): A list of light spot index arrays for each Tx–Rx pair.
            Each array has shape (num_scatterers, 3), where each row represents 
            [row_index, col_index, delay_index].
        Tx_positions (np.ndarray): The 3D positions of transmitters, shape (num_pairs, 3).
        Rx_positions (np.ndarray): The 3D positions of receivers, shape (num_pairs, 3).
        Tx_orientations (np.ndarray): The orientations of transmitters in Sionna form, shape (num_pairs, 3).
        num_row (int): Number of vertical grid points in the light spot map.
        num_col (int): Number of horizontal grid points in the light spot map.
        delta_f (float): Subcarrier spacing in Hz.
        num_subcarrier (int): Total number of subcarriers.

    Returns:
        list[np.ndarray]: A list containing the estimated scatterer positions for each Tx–Rx pair.
            Each element has shape (num_scatterers, 3), representing 3D scatterer coordinates.

    Notes:
        - This function converts light spot indices (representations of θ, φ, τ in CSI space)
          into actual scatterer coordinates in 3D space.
        - The conversion process includes two main steps:
            1. Mapping the grid indices to physical parameters (θ, φ, τ).
            2. Converting (θ, φ, τ) to Cartesian scatterer coordinates using geometric constraints.
        - The resulting positions correspond to points on ellipsoids defined by the given Tx–Rx pair.
    """   
    scatterer_positions=[]
    for id in range(len(light_spot_indices)):

        light_spot_index=light_spot_indices[id]
        tx_pos = Tx_positions[id]
        rx_pos = Rx_positions[id]
        num_scatterers = light_spot_index.shape[0]
        scatterer_positions.append(np.zeros((num_scatterers,3)))
        # Compute the distances and angles for each scatterer
        for i in range(num_scatterers):
            index=light_spot_index[i]
            theta,phi,tau=light_spot_index2theta_phi_tau(index,num_row,num_col,delta_f,num_subcarrier)
            scatterer_pos=theta_phi_tau2scatterer_pos(tx_pos,rx_pos,theta,phi,tau)
            scatter_positions[id][i]=scatterer_pos
    return scatter_positions

def locate_scatterers_in_CSI(scatterer_positions, Tx_positions, Rx_positions, Tx_orientations, num_row,num_col,delta_f,num_subcarrier):
    """Locate the corresponding CSI light spot indices of given 3D scatterers.

    Args:
        scatterer_positions (list[np.ndarray]): A list of scatterer position arrays for each Tx–Rx pair.
            Each array has shape (num_scatterers, 3), where each row represents the 3D coordinates of one scatterer.
        Tx_positions (np.ndarray): The 3D positions of transmitters, shape (num_pairs, 3).
        Rx_positions (np.ndarray): The 3D positions of receivers, shape (num_pairs, 3).
        Tx_orientations (np.ndarray): The orientations of transmitters in Sionna format, shape (num_pairs, 3).
        num_row (int): Number of vertical grid points in the light spot map.
        num_col (int): Number of horizontal grid points in the light spot map.
        delta_f (float): Subcarrier spacing in Hz.
        num_subcarrier (int): Total number of subcarriers.

    Returns:
        list[np.ndarray]: A list of index arrays for each Tx–Rx pair.
            Each array has shape (num_scatterers, 3), where each row represents
            [row_index, col_index, delay_index] corresponding to one scatterer.

    Notes:
        - This function performs the inverse operation of scatterer localization: 
          it maps known 3D scatterer positions back to their corresponding indices 
          (θ, φ, τ) in the CSI light spot domain.
        - The process involves two main steps:
            1. Computing the propagation delay τ of each scatterer based on geometric distances.
            2. Converting (θ, φ, τ) to discrete grid indices using `theta_phi_tau2light_spot_index`.
        - The computed indices can be used to highlight or visualize scatterer contributions 
          in the CSI-derived light spot map.
    """
    indices=[]
    taus=[np.zeros(len(scatterer_positions[i])) for i in range(len(scatterer_positions))]
    for i in range(len(Tx_positions)):
        for j in range(len(scatterer_positions[i])):
            taus[i][j]=(np.linalg.norm(np.reshape(Tx_positions[i],(3,)) - scatterer_positions[i][j])
                        + np.linalg.norm(np.reshape(Rx_positions[i],(3,)) - scatterer_positions[i][j])
                        - np.linalg.norm(Tx_positions[i] - Rx_positions[i])) / 3e8
    for i in range(len(scatterer_positions)):
        scatterer_pos=scatterer_positions[i]
        indices.append(np.zeros((len(scatterer_pos),3)))
        for j in range(len(scatterer_pos)):
            theta,phi=compute_AoD(np.squeeze(Tx_positions[i]),np.squeeze(scatterer_pos[j]),Tx_orientations[i])
            index=theta_phi_tau2light_spot_index(theta,phi,taus[i][j],num_row,num_col,delta_f,num_subcarrier)
            indices[i][j]=index
    return indices


if __name__ == "__main__":
    data=np.load("/home/jingpeng/graduation_project/Channel_Simulation/dataset/sim_results20251023.npy", allow_pickle=True)
    print(data.item().keys())
    CSI=data.item().get("CSI")
    scatter_positions=data.item().get("scatter_positions")
    Tx_positions=data.item().get("Tx_positions")
    Rx_positions=data.item().get("Rx_positions")
    Tx_orientations=data.item().get("Tx_orientations")
    amplitude=data.item().get("amplitudes")


    indices=locate_scatterers_in_CSI(scatter_positions,Tx_positions,Rx_positions,Tx_orientations,16,16,100e6/1024,1024)

    print(indices)

    # taus=[np.zeros(len(scatter_positions[i])) for i in range(len(scatter_positions))]
    # for i in range(len(Tx_positions)):
    #     for j in range(len(scatter_positions[i])):
    #         taus[i][j]=(np.linalg.norm(np.reshape(Tx_positions[i],(3,)) - scatter_positions[i][j])
    #                     + np.linalg.norm(np.reshape(Rx_positions[i],(3,)) - scatter_positions[i][j])
    #                     - np.linalg.norm(Tx_positions[i] - Rx_positions[i])) / 3e8

    # TestCSI=[]
    # for i in range(8):
    #     TestCSI.append(reconstruct_CSI_from_scatterers_and_amplitudes(Tx_positions[i],
    #                                                 scatter_positions[i],
    #                                                 amplitude[i],
    #                                                 Tx_orientations[i],
    #                                                 num_subcarriers=1024,
    #                                                 Tx_ant_rows=16,
    #                                                 Tx_ant_cols=16,
    #                                                 carrier_frequency=3e9,
    #                                                 bandwidth=100e6,
    #                                                 speed_of_light=3e8,
    #                                                 taus=taus[i]))
    # save_results={"CSI":TestCSI}
    #np.save("/home/jingpeng/graduation_project/Channel_Simulation/debug_file/test_reconstructed_CSI.npy", save_results)
    #locate_scatterers_in_CSI(CSI, scatter_positions, Tx_positions, Rx_positions, Tx_orientations)