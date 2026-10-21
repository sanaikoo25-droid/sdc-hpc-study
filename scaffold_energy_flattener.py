import random
import numpy as np
import json

NN_dG = {
    'AA': -1.00, 'TT': -1.00, 'AT': -0.88, 'TA': -0.58,
    'CA': -1.45, 'TG': -1.45, 'GT': -1.44, 'AC': -1.44,
    'CT': -1.28, 'AG': -1.28, 'GA': -1.30, 'TC': -1.30,
    'CG': -2.17, 'GC': -2.24, 'GG': -1.84, 'CC': -1.84
}

def calc_duplex_dG(seq_top, seq_bottom):
    dG = 0.0
    for i in range(len(seq_top) - 1):
        p_top = seq_top[i:i+2]
        p_bot = seq_bottom[i:i+2]
        if (p_top[0] + p_bot[0] in ['AT','TA','GC','CG']) and \
           (p_top[1] + p_bot[1] in ['AT','TA','GC','CG']):
            dG += NN_dG.get(p_top, -1.2)
        else:
            dG += +0.85
    return dG

def optimize_domain(scaffold_seq, target_dG=-18.5, max_iters=3000):
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    curr_seq = [comp[b] for b in scaffold_seq]
    best_seq = list(curr_seq)
    best_loss = abs(calc_duplex_dG(scaffold_seq, "".join(curr_seq)) - target_dG)

    for _ in range(max_iters):
        idx = random.randint(0, len(scaffold_seq) - 1)
        orig = curr_seq[idx]
        curr_seq[idx] = random.choice(['A', 'C', 'G', 'T'])
        loss = abs(calc_duplex_dG(scaffold_seq, "".join(curr_seq)) - target_dG)
        if loss < best_loss:
            best_loss = loss
            best_seq = list(curr_seq)
        else:
            curr_seq[idx] = orig
    return "".join(best_seq), calc_duplex_dG(scaffold_seq, "".join(best_seq))

if __name__ == "__main__":
    np.random.seed(42)
    bases = ['A', 'C', 'G', 'T']
    results = {}
    print(f"{'Domain Pos':<12} | {'Orig dG (kcal/mol)':<20} | {'Tuned dG (kcal/mol)':<20} | {'Target dG':<12}", flush=True)
    print("-" * 70, flush=True)

    for pos in range(25):
        scaff_seq = "".join(np.random.choice(bases, size=24, p=[0.2, 0.3, 0.3, 0.2]))
        comp_seq = "".join([{'A':'T','T':'A','C':'G','G':'C'}[b] for b in scaff_seq])
        orig_dG = calc_duplex_dG(scaff_seq, comp_seq)

        opt_seq, final_dG = optimize_domain(scaff_seq, target_dG=-18.5)
        results[pos] = {
            "scaffold_seq": scaff_seq,
            "optimized_tile_seq": opt_seq,
            "original_dG": orig_dG,
            "tuned_dG": final_dG
        }
        print(f"{pos:<12} | {orig_dG:<20.2f} | {final_dG:<20.2f} | {-18.5:<12.1f}", flush=True)

    with open("flattened_domains.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nOptimization complete. Flattened sequences saved to flattened_domains.json", flush=True)
