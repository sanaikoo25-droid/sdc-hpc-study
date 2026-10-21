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

def analyze_base_pairing(frames):
    frames_nm = frames * 0.8518
    num_frames = len(frames_nm)
    
    # 4 positions x 24 bp duplexes = 96 base pairs total
    # Scaffold: 0 to 95
    # Tile 1 duplex: 108 to 131 (paired with scaffold 0 to 23)
    # Tile 2 duplex: 156 to 179 (paired with scaffold 24 to 47)
    # Tile 3 duplex: 204 to 227 (paired with scaffold 48 to 71)
    # Tile 4 duplex: 252 to 275 (paired with scaffold 72 to 95)
    
    tile_duplex_ranges = [
        (range(0, 24), range(108, 132)),
        (range(24, 48), range(156, 180)),
        (range(48, 72), range(204, 228)),
        (range(72, 96), range(252, 276))
    ]
    
    print(f"{'Position':<10} | {'Domain':<12} | {'Mean Duplex RMSD (nm)':<24} | {'Intact Fraction (%)':<20}")
    print("-" * 72)
    
    for pos_idx, (scaff_range, tile_range) in enumerate(tile_duplex_ranges):
        intact_frames = 0
        mean_dists = []
        for f in range(num_frames):
            scaff_pts = frames_nm[f, scaff_range, :]
            tile_pts = frames_nm[f, tile_range, :]
            
            # Distance between paired base positions (Watson-Crick contact cutoff ~ 1.2 nm in oxDNA)
            pair_dists = np.linalg.norm(scaff_pts - tile_pts, axis=1)
            mean_d = np.mean(pair_dists)
            mean_dists.append(mean_d)
            if np.all(pair_dists < 1.4):  # duplex contact maintained
                intact_frames += 1
                
        frac = (intact_frames / num_frames) * 100.0
        pos_names = ["A", "B", "C", "D"]
        print(f"Pos {pos_names[pos_idx]:<6} | 24-nt core   | {np.mean(mean_dists):<24.2f} | {frac:<20.1f}")

if __name__ == "__main__":
    frames = parse_oxdna_trajectory("sdc_trajectory.dat")
    analyze_base_pairing(frames)
