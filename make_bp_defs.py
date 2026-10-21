with open("hb_bp.txt", "w") as f:
    f.write("{\n")
    f.write("order_parameter = hydrogen_bonding\n")
    f.write("name = sdc_duplex_contacts\n")
    f.write("file = bp_fraction.dat\n")
    
    pairs = []
    # Pos A: Scaffold 0..23 <-> Tile 1 core (131 down to 108)
    for s_idx, t_idx in zip(range(0, 24), reversed(range(108, 132))):
        pairs.append(f"{s_idx},{t_idx}")
    # Pos B: Scaffold 24..47 <-> Tile 2 core (179 down to 156)
    for s_idx, t_idx in zip(range(24, 48), reversed(range(156, 180))):
        pairs.append(f"{s_idx},{t_idx}")
    # Pos C: Scaffold 48..71 <-> Tile 3 core (227 down to 204)
    for s_idx, t_idx in zip(range(48, 72), reversed(range(204, 228))):
        pairs.append(f"{s_idx},{t_idx}")
    # Pos D: Scaffold 72..95 <-> Tile 4 core (275 down to 252)
    for s_idx, t_idx in zip(range(72, 96), reversed(range(252, 276))):
        pairs.append(f"{s_idx},{t_idx}")
        
    f.write("pairs = " + " ".join(pairs) + "\n")
    f.write("}\n")

print(f"Generated hb_bp.txt with file = bp_fraction.dat ({len(pairs)} pairs).")
