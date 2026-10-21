import sys

def convert_oxdna_to_pdb(top_file, conf_file, pdb_file):
    # Read topology
    with open(top_file, 'r') as f:
        header = f.readline().split()
        total_nt = int(header[0])
        total_strands = int(header[1])
        top_data = []
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                top_data.append((int(parts[0]), parts[1]))  # strand_id, base

    # Read configuration coordinates
    coords = []
    with open(conf_file, 'r') as f:
        for line in f:
            line_str = line.strip()
            if line_str.startswith("t =") or line_str.startswith("b =") or line_str.startswith("E ="):
                continue
            parts = line_str.split()
            if len(parts) >= 3:
                try:
                    # 1 oxDNA unit = 0.8518 nm = 8.518 Angstroms
                    x = float(parts[0]) * 8.518
                    y = float(parts[1]) * 8.518
                    z = float(parts[2]) * 8.518
                    coords.append((x, y, z))
                except ValueError:
                    continue

    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    with open(pdb_file, 'w') as f:
        f.write("HEADER    SCAFFOLDED DNA COMPUTER 4-POSITION COMPLEX\n")
        for i in range(min(len(top_data), len(coords))):
            strand_id, base = top_data[i]
            x, y, z = coords[i]
            chain = chain_letters[(strand_id - 1) % len(chain_letters)]
            atom_num = (i + 1) % 100000
            res_num = (i + 1) % 10000
            
            # PDB ATOM record format
            line = f"ATOM  {atom_num:5d}  P   D{base:1s}  {chain:1s}{res_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           P  \n"
            f.write(line)
        f.write("END\n")

    print(f"PDB successfully created: {pdb_file} ({len(coords)} nucleotides written)")

if __name__ == "__main__":
    top = "sdc_4pos.top"
    conf = "sdc_lastconf.dat"
    out = "sdc_final_structure.pdb"
    convert_oxdna_to_pdb(top, conf, out)
