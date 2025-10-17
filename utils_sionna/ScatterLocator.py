import numpy as np

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
            d_tx_scatter = np.linalg.norm(scatter_pos - tx_pos)
            d_scatter_rx = np.linalg.norm(rx_pos - scatter_pos)
            total_distance = d_tx_scatter + d_scatter_rx

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
    data=np.load("/home/jingpeng/graduation_project/Channel_Simulation/dataset/sim_results20251017.npy", allow_pickle=True)
    CSI=data.item().get("CSI")
    scatter_positions=data.item().get("scatter_positions")
    Tx_positions=data.item().get("Tx_positions")
    Rx_positions=data.item().get("Rx_positions")
    Tx_orientations=data.item().get("Tx_orientations")
    locate_scatterers_in_CSI(CSI, scatter_positions, Tx_positions, Rx_positions, Tx_orientations)