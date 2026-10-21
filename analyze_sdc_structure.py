import numpy as np
import sys

def parse_oxdna_trajectory(traj_file, total_particles=288):
    """Parses coordinates from oxDNA trajectory file."""
    frames = []
    current_coords = []
    
    with open(traj_file, 'r') as f:
        reading_coords = False
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                if current_coords:
                    frames.append(np.array(current_coords))
                    current_coords = []
                reading_coords = False
                continue
            
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    pos = [float(parts[0]), float(parts[1]), float(parts[2])]
                    current_coords.append(pos)
                except ValueError:
                    continue
                    
        if current_coords:
            frames.append(np.array(current_coords))
            
    return np.array(frames)

def analyze_system(frames):
    # Total particles = 288
    # Scaffold: 0 to 95 (96 nt)
    # Tile 1: 96 to 143 (48 nt) -> left arm: 96-107, core: 108-131, right arm: 132-143
    # Tile 2: 144 to 191
    # Tile 3: 192 to 239
    # Tile 4: 240 to 287
    
    num_frames = len(frames)
    print(f"Loaded {num_frames} frames for structural analysis.")
    if num_frames == 0:
        return

    # 1. Scaffold end-to-end extension (nt 0 to nt 95)
    scaff_dists = np.linalg.norm(frames[:, 95, :] - frames[:, 0, :], axis=1) * 0.8518  # in nm
    print(f"Scaffold End-to-End Extension: {np.mean(scaff_dists):.2f} +/- {np.std(scaff_dists):.2f} nm")

    # 2. Compute-arm spatial RMSF (flexibility & availability)
    arm_indices = list(range(96, 108)) + list(range(132, 144))
    arm_coords = frames[:, arm_indices, :]
    mean_pos = np.mean(arm_coords, axis=0)
    rmsf = np.sqrt(np.mean(np.sum((arm_coords - mean_pos)**2, axis=-1), axis=0)) * 0.8518
    print(f"Flanking Compute-Arm Mean RMSF: {np.mean(rmsf):.2f} nm (Free exploration range)")

    # 3. Inter-tile duplex center-of-mass spacing
    # Core tile 1 (108 to 131) vs Core tile 2 (156 to 179)
    t1_com = np.mean(frames[:, 108:132, :], axis=1)
    t2_com = np.mean(frames[:, 156:180, :], axis=1)
    inter_tile_dist = np.linalg.norm(t2_com - t1_com, axis=1) * 0.8518
    print(f"Adjacent Inter-Tile Distance: {np.mean(inter_tile_dist):.2f} +/- {np.std(inter_tile_dist):.2f} nm")

if __name__ == "__main__":
    traj_path = "sdc_trajectory.dat"
    try:
        data = parse_oxdna_trajectory(traj_path)
        analyze_system(data)
    except FileNotFoundError:
        print(f"Trajectory file {traj_path} not found yet. Check if MD is writing.")
