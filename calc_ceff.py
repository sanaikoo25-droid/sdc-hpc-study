import numpy as np

def parse_oxdna_trajectory(traj_file):
    frames = []
    current_coords = []
    with open(traj_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                if current_coords:
                    frames.append(np.array(current_coords))
                    current_coords = []
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    current_coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        if current_coords:
            frames.append(np.array(current_coords))
    return np.array(frames)

def compute_effective_concentration(frames):
    # oxDNA unit to meters: 1 unit = 0.8518 nm = 8.518e-10 m
    # Tile 1 Right Arm (132-143) to Tile 2 Left Arm (144-155)
    frames_nm = frames * 0.8518
    num_frames = len(frames_nm)
    
    arm1_tip = frames_nm[:, 143, :]  # Terminal nucleotide of Tile 1 right arm
    arm2_tip = frames_nm[:, 144, :]  # Terminal nucleotide of Tile 2 left arm
    
    dists_nm = np.linalg.norm(arm1_tip - arm2_tip, axis=-1)
    
    # Calculate radius of gyration / exploration sphere
    r_mean_nm = np.mean(dists_nm)
    r_std_nm = np.std(dists_nm)
    
    # Effective volume V = (4/3) * pi * (r_mean)^3 in Liters
    r_meters = r_mean_nm * 1e-9
    volume_liters = (4.0 / 3.0) * np.pi * (r_meters**3) * 1000.0
    
    # C_eff = 1 molecule / (N_A * Volume)
    N_A = 6.022e23
    c_eff_molar = 1.0 / (N_A * volume_liters)
    c_eff_micromolar = c_eff_molar * 1e6
    
    print("--- Effective Local Concentration Analysis ---")
    print(f"Mean Inter-Arm Terminal Separation: {r_mean_nm:.2f} +/- {r_std_nm:.2f} nm")
    print(f"Equivalent Spherical Volume: {volume_liters:.3e} L")
    print(f"Effective Local Concentration (C_eff): {c_eff_micromolar:.3f} uM")
    print(f"Bulk Tile Excess in Paper: 1.0 uM (10x over 100 nM scaffold)")

if __name__ == "__main__":
    frames = parse_oxdna_trajectory("sdc_trajectory.dat")
    compute_effective_concentration(frames)
