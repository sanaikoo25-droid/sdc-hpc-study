import numpy as np

# oxDNA distance unit = 0.8518 nm
RISE_BP = 0.34 / 0.8518        # ~0.399
RADIUS = 1.0 / 0.8518          # ~1.174
BP_PER_TURN = 10.5
TWIST_PER_BP = (2.0 * np.pi) / BP_PER_TURN
FENE_R0 = 0.7525               # Equilibrium bonded distance in oxDNA

BOX_SIZE = 60.0

def generate_sdc_system():
    scaffold_seq = "AGCTACGTACGTTAGCTAGCTAGC" * 4  # 96 nt
    
    tile_seqs = [
        "ATCGATCGATCG" + "GCTAGCTAGCTAACGTACGTAGCT" + "CGTACGTACGTA",
        "CGTACGTACGTA" + "GCTAGCTAGCTAACGTACGTAGCT" + "ATCGATCGATCG",
        "ATCGATCGATCG" + "GCTAGCTAGCTAACGTACGTAGCT" + "CGTACGTACGTA",
        "CGTACGTACGTA" + "GCTAGCTAGCTAACGTACGTAGCT" + "ATCGATCGATCG"
    ]

    top_lines = []
    dat_lines = []

    total_nt = len(scaffold_seq) + sum(len(s) for s in tile_seqs)
    total_strands = 1 + len(tile_seqs)
    top_lines.append(f"{total_nt} {total_strands}\n")

    dat_lines.append(f"t = 0\n")
    dat_lines.append(f"b = {BOX_SIZE} {BOX_SIZE} {BOX_SIZE}\n")
    dat_lines.append(f"E = 0 0 0\n")

    current_idx = 0

    # 1. Generate Scaffold Strand (Strand 1)
    strand_id = 1
    z_offset = -(len(scaffold_seq) * RISE_BP) / 2.0

    for i, base in enumerate(scaffold_seq):
        theta = i * TWIST_PER_BP
        pos_x = RADIUS * np.cos(theta)
        pos_y = RADIUS * np.sin(theta)
        pos_z = z_offset + i * RISE_BP

        a1 = np.array([-np.cos(theta), -np.sin(theta), 0.0])
        a3 = np.array([0.0, 0.0, 1.0])
        a1 /= np.linalg.norm(a1)
        a3 /= np.linalg.norm(a3)

        n3 = current_idx - 1 if i > 0 else -1
        n5 = current_idx + 1 if i < len(scaffold_seq) - 1 else -1
        top_lines.append(f"{strand_id} {base} {n3} {n5}\n")

        dat_lines.append(
            f"{pos_x:.6f} {pos_y:.6f} {pos_z:.6f} "
            f"{a1[0]:.6f} {a1[1]:.6f} {a1[2]:.6f} "
            f"{a3[0]:.6f} {a3[1]:.6f} {a3[2]:.6f} "
            f"0 0 0 0 0 0\n"
        )
        current_idx += 1

    # 2. Generate 4 Compute Tile Strands (Strands 2 through 5)
    for t_idx, t_seq in enumerate(tile_seqs):
        strand_id += 1
        pos_start_idx = t_idx * 24

        # Compute reference position of the first duplex base (j=12)
        theta_duplex_start = pos_start_idx * TWIST_PER_BP
        x_dup_start = -RADIUS * np.cos(theta_duplex_start)
        y_dup_start = -RADIUS * np.sin(theta_duplex_start)
        z_dup_start = z_offset + pos_start_idx * RISE_BP

        # Compute reference position of the last duplex base (j=35)
        pos_end_idx = pos_start_idx + 23
        theta_duplex_end = pos_end_idx * TWIST_PER_BP
        x_dup_end = -RADIUS * np.cos(theta_duplex_end)
        y_dup_end = -RADIUS * np.sin(theta_duplex_end)
        z_dup_end = z_offset + pos_end_idx * RISE_BP

        for j, base in enumerate(t_seq):
            if j < 12:
                # Left single-stranded compute arm extending along radial vector
                arm_steps = 12 - j
                radial_dir = np.array([-np.cos(theta_duplex_start), -np.sin(theta_duplex_start), 0.0])
                pos_x = x_dup_start + radial_dir[0] * (arm_steps * FENE_R0)
                pos_y = y_dup_start + radial_dir[1] * (arm_steps * FENE_R0)
                pos_z = z_dup_start - arm_steps * 0.1
                a1 = np.array([radial_dir[0], radial_dir[1], 0.0])
                a3 = np.array([0.0, 0.0, -1.0])

            elif j < 36:
                # 24-nt binding domain hybridized to scaffold position
                scaff_pos = pos_start_idx + (j - 12)
                theta = scaff_pos * TWIST_PER_BP
                pos_x = -RADIUS * np.cos(theta)
                pos_y = -RADIUS * np.sin(theta)
                pos_z = z_offset + scaff_pos * RISE_BP
                a1 = np.array([np.cos(theta), np.sin(theta), 0.0])
                a3 = np.array([0.0, 0.0, -1.0])

            else:
                # Right single-stranded compute arm extending outward
                arm_steps = j - 35
                radial_dir = np.array([-np.cos(theta_duplex_end), -np.sin(theta_duplex_end), 0.0])
                pos_x = x_dup_end + radial_dir[0] * (arm_steps * FENE_R0)
                pos_y = y_dup_end + radial_dir[1] * (arm_steps * FENE_R0)
                pos_z = z_dup_end + arm_steps * 0.1
                a1 = np.array([radial_dir[0], radial_dir[1], 0.0])
                a3 = np.array([0.0, 0.0, -1.0])

            a1 /= np.linalg.norm(a1)
            a3 /= np.linalg.norm(a3)

            n3 = current_idx - 1 if j > 0 else -1
            n5 = current_idx + 1 if j < len(t_seq) - 1 else -1
            top_lines.append(f"{strand_id} {base} {n3} {n5}\n")

            dat_lines.append(
                f"{pos_x:.6f} {pos_y:.6f} {pos_z:.6f} "
                f"{a1[0]:.6f} {a1[1]:.6f} {a1[2]:.6f} "
                f"{a3[0]:.6f} {a3[1]:.6f} {a3[2]:.6f} "
                f"0 0 0 0 0 0\n"
            )
            current_idx += 1

    with open("sdc_4pos.top", "w") as f:
        f.writelines(top_lines)
    with open("sdc_4pos.dat", "w") as f:
        f.writelines(dat_lines)

    print("Regenerated sdc_4pos.top and sdc_4pos.dat with consistent FENE bond geometry.")

if __name__ == "__main__":
    generate_sdc_system()
