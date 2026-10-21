import numpy as np

def convert_to_valid_pdb(top_file, conf_file, out_pdb):
    # Parse topology
    strands = []
    bases = []
    with open(top_file, 'r') as f:
        header = f.readline().split()
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                strands.append(int(parts[0]))
                bases.append(parts[1])

    # Parse coordinates
    coords = []
    with open(conf_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    # oxDNA distance unit: 1 unit = 0.8518 nm = 8.518 Angstroms
                    x = float(parts[0]) * 8.518
                    y = float(parts[1]) * 8.518
                    z = float(parts[2]) * 8.518
                    coords.append([x, y, z])
                except ValueError:
                    continue

    coords = np.array(coords)
    # Center system at (0, 0, 0)
    coords = coords - np.mean(coords, axis=0)

    chain_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    
    with open(out_pdb, 'w') as f:
        f.write("HEADER    SCAFFOLDED DNA COMPUTER 4-POSITION COMPLEX\n")
        for i in range(len(coords)):
            s_id = strands[i]
            base = bases[i]
            x, y, z = coords[i]
            chain = chain_map.get(s_id, 'X')
            atom_idx = i + 1
            res_idx = (i % 96) + 1
            
            # Exact standard fixed-column PDB format: %8.3f with proper spacing
            f.write(
                f"ATOM  {atom_idx:5d}  P   D{base:1s}  {chain:1s}{res_idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           P  \n"
            )
        f.write("END\n")

    print(f"Generated correctly formatted, centered PDB: {out_pdb}")

if __name__ == "__main__":
    convert_to_valid_pdb("sdc_4pos.top", "sdc_lastconf.dat", "sdc_final_structure.pdb")
