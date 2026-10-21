import numpy as np

BOX_SIZE_NM = 60.0 * 0.8518  # 51.108 nm

def unwrap_and_write_pdb(top_file, conf_file, out_pdb):
    strands = []
    bases = []
    with open(top_file, 'r') as f:
        f.readline()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                strands.append(int(parts[0]))
                bases.append(parts[1])

    coords = []
    with open(conf_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    # nm units
                    x = float(parts[0]) * 0.8518
                    y = float(parts[1]) * 0.8518
                    z = float(parts[2]) * 0.8518
                    coords.append([x, y, z])
                except ValueError:
                    continue

    coords = np.array(coords)
    box = BOX_SIZE_NM

    # Scaffold is Strand 1 (indices 0 to 95)
    scaff_coords = coords[:96]
    scaff_com = np.mean(scaff_coords, axis=0)

    # Unwrap tiles (Strands 2 through 5) relative to their binding positions on Scaffold
    # Tile 1 (Chain B): indices 96 to 143  <-> pairs with scaff 0..23
    # Tile 2 (Chain C): indices 144 to 191 <-> pairs with scaff 24..47
    # Tile 3 (Chain D): indices 192 to 239 <-> pairs with scaff 48..71
    # Tile 4 (Chain E): indices 240 to 287 <-> pairs with scaff 72..95
    tile_ranges = [
        (96, 144, 0, 24),
        (144, 192, 24, 48),
        (192, 240, 48, 72),
        (240, 288, 72, 96)
    ]

    for t_start, t_end, s_start, s_end in tile_ranges:
        target_scaff_com = np.mean(coords[s_start:s_end], axis=0)
        tile_com = np.mean(coords[t_start:t_end], axis=0)
        
        # Calculate periodic shift to bring tile COM closest to its scaffold domain
        shift = box * np.round((tile_com - target_scaff_com) / box)
        coords[t_start:t_end] -= shift

    # Center entire unwrapped complex at (0, 0, 0) in Angstroms
    coords_angstrom = (coords - np.mean(coords, axis=0)) * 10.0

    chain_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    with open(out_pdb, 'w') as f:
        f.write("HEADER    SCAFFOLDED DNA COMPUTER 4-POSITION UNWRAPPED\n")
        for i in range(len(coords_angstrom)):
            s_id = strands[i]
            base = bases[i]
            x, y, z = coords_angstrom[i]
            chain = chain_map.get(s_id, 'X')
            atom_idx = i + 1
            res_idx = (i % 96) + 1
            f.write(
                f"ATOM  {atom_idx:5d}  P   D{base:1s}  {chain:1s}{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           P  \n"
            )
        f.write("END\n")

    print(f"Generated PBC-unwrapped, assembled PDB: {out_pdb}")

if __name__ == "__main__":
    unwrap_and_write_pdb("sdc_4pos.top", "sdc_lastconf.dat", "sdc_final_structure.pdb")
