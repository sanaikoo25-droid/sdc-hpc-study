import numpy as np

BOX = 60.0 * 0.8518  # 51.108 nm

def unwrap_strand_continuous(coords, box=BOX):
    """Unwraps a single strand so each subsequent nucleotide is within box/2 of the previous."""
    unwrapped = [coords[0].copy()]
    for i in range(1, len(coords)):
        prev = unwrapped[-1]
        curr = coords[i].copy()
        shift = box * np.round((curr - prev) / box)
        unwrapped.append(curr - shift)
    return np.array(unwrapped)

def build_assembled_complex(top_file, conf_file, out_pdb):
    # Parse topology
    strands = []
    bases = []
    with open(top_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                strands.append(int(parts[0]))
                bases.append(parts[1])

    # Parse raw coordinates (in nm)
    raw_coords = []
    with open(conf_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    raw_coords.append([float(parts[0]) * 0.8518, float(parts[1]) * 0.8518, float(parts[2]) * 0.8518])
                except ValueError:
                    continue

    raw_coords = np.array(raw_coords)
    box = BOX

    # 1. Fully unwrap the Scaffold (Strand 1: indices 0 to 95)
    scaff_unwrapped = unwrap_strand_continuous(raw_coords[0:96], box=box)
    final_coords = np.zeros_like(raw_coords)
    final_coords[0:96] = scaff_unwrapped

    # 2. Align and unwrap each tile (Strands 2 through 5, each 48 nt)
    # Tile 1: 96..143  (core 108..131 pairs with scaff 23 down to 0)
    # Tile 2: 144..191 (core 156..179 pairs with scaff 47 down to 24)
    # Tile 3: 192..239 (core 204..227 pairs with scaff 71 down to 48)
    # Tile 4: 240..287 (core 252..275 pairs with scaff 95 down to 72)
    tile_definitions = [
        (96,  144, 120, 11),  # Tile 1: anchor core mid-nt 120 to scaff nt 11
        (144, 192, 168, 35),  # Tile 2: anchor core mid-nt 168 to scaff nt 35
        (192, 240, 216, 59),  # Tile 3: anchor core mid-nt 216 to scaff nt 59
        (240, 288, 264, 83)   # Tile 4: anchor core mid-nt 264 to scaff nt 83
    ]

    for t_start, t_end, t_anchor_idx, s_partner_idx in tile_definitions:
        tile_raw = raw_coords[t_start:t_end]
        # First unwrap the tile along its own 48-nt backbone
        tile_contiguous = unwrap_strand_continuous(tile_raw, box=box)
        
        # Determine shift between the tile's anchor base and its scaffold partner
        t_anchor_pos = tile_contiguous[t_anchor_idx - t_start]
        s_partner_pos = scaff_unwrapped[s_partner_idx]
        
        global_shift = box * np.round((t_anchor_pos - s_partner_pos) / box)
        final_coords[t_start:t_end] = tile_contiguous - global_shift

    # Center entire assembled molecular complex at origin (in Angstroms)
    center = np.mean(final_coords, axis=0)
    final_coords_angstrom = (final_coords - center) * 10.0

    chain_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    with open(out_pdb, 'w') as f:
        f.write("HEADER    SCAFFOLDED DNA COMPUTER 4-POSITION UNWRAPPED\n")
        for i in range(len(final_coords_angstrom)):
            s_id = strands[i]
            base = bases[i]
            x, y, z = final_coords_angstrom[i]
            chain = chain_map.get(s_id, 'X')
            atom_idx = i + 1
            res_idx = (i % 96) + 1
            f.write(
                f"ATOM  {atom_idx:5d}  P   D{base:1s}  {chain:1s}{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           P  \n"
            )
        f.write("END\n")

    print(f"Generated clean unwrapped PDB: {out_pdb}")

if __name__ == "__main__":
    build_assembled_complex("sdc_4pos.top", "sdc_lastconf.dat", "sdc_final_structure.pdb")
