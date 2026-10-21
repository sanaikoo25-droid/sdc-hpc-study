import numpy as np

BOX_SIZE_NM = 60.0 * 0.8518  # 51.108 nm

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

def pbc_dist(p1, p2, box=BOX_SIZE_NM):
    diff = p1 - p2
    diff = diff - box * np.round(diff / box)
    return np.linalg.norm(diff, axis=-1)

def analyze_corrected():
    frames = parse_oxdna_trajectory("sdc_trajectory.dat") * 0.8518
    num_frames = len(frames)
    
    # Correct antiparallel indexing:
    # Scaffold 0..23 pairs with Tile 1 (131 down to 108)
    # Scaffold 24..47 pairs with Tile 2 (179 down to 156)
    # Scaffold 48..71 pairs with Tile 3 (227 down to 204)
    # Scaffold 72..95 pairs with Tile 4 (275 down to 252)
    tile_duplex_ranges = [
        (list(range(0, 24)), list(reversed(range(108, 132)))),
        (list(range(24, 48)), list(reversed(range(156, 180)))),
        (list(range(48, 72)), list(reversed(range(204, 228)))),
        (list(range(72, 96)), list(reversed(range(252, 276))))
    ]
    
    print(f"{'Position':<10} | {'Domain':<12} | {'Mean Duplex Pair Dist (nm)':<28} | {'Intact Fraction (%)':<20}")
    print("-" * 75)
    
    pos_names = ["A", "B", "C", "D"]
    for pos_idx, (scaff_idx, tile_idx) in enumerate(tile_duplex_ranges):
        intact_frames = 0
        mean_dists = []
        for f in range(num_frames):
            s_pts = frames[f, scaff_idx, :]
            t_pts = frames[f, tile_idx, :]
            
            dists = pbc_dist(s_pts, t_pts)
            mean_dists.append(np.mean(dists))
            # Standard Watson-Crick center-of-mass contact in oxDNA: < 1.35 nm
            if np.all(dists < 1.35):
                intact_frames += 1
                
        frac = (intact_frames / num_frames) * 100.0
        print(f"Pos {pos_names[pos_idx]:<6} | 24-nt core   | {np.mean(mean_dists):<28.2f} | {frac:<20.1f}")

if __name__ == "__main__":
    analyze_corrected()
