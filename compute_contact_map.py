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

def compute_arm_contacts(frames, cutoff_nm=2.0):
    # oxDNA distance to nm
    frames_nm = frames * 0.8518
    num_frames = len(frames_nm)
    
    # Tile 1 Right Compute Arm: particles 132 to 143
    # Tile 2 Left Compute Arm: particles 144 to 155
    arm1_right = frames_nm[:, 132:144, :]
    arm2_left = frames_nm[:, 144:156, :]
    
    min_dists = []
    contact_count = 0
    
    for f in range(num_frames):
        # Compute all pairwise distances between the two adjacent 12-nt arms
        diff = arm1_right[f, :, np.newaxis, :] - arm2_left[f, np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        min_d = np.min(dists)
        min_dists.append(min_d)
        if min_d <= cutoff_nm:
            contact_count += 1
            
    contact_prob = (contact_count / num_frames) * 100.0
    mean_min_dist = np.mean(min_dists)
    
    print(f"--- 3D Compute Arm Encounter Metrics (Adjacent Tiles 1 & 2) ---")
    print(f"Total Frames Analyzed: {num_frames}")
    print(f"Mean Minimum Inter-Arm Distance: {mean_min_dist:.2f} nm")
    print(f"Encounter Probability (within {cutoff_nm} nm): {contact_prob:.1f}%")
    
if __name__ == "__main__":
    frames = parse_oxdna_trajectory("sdc_trajectory.dat")
    compute_arm_contacts(frames)
