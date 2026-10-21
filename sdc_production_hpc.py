import numpy as np
import json
import sys
from multiprocessing import Pool

R_KCAL = 0.0019872

# Experimentally and oxDNA-calibrated parameters
C_EFF_MOLAR = 63.368e-6      # 63.37 uM from oxDNA trajectory analysis
BULK_TILE_EXCESS = 1.0e-6    # 1.0 uM bulk concentration
K_ON = 1e6
COMPUTE_MATCH_DG_REF = -9.2  # kcal/mol at reference
MISMATCH_PENALTY = +3.5      # kcal/mol enthalpic penalty

def get_temperature(sim_time, duration):
    frac = min(1.0, sim_time / duration)
    if frac < 0.15:
        return 80.0 - (80.0 - 58.0) * (frac / 0.15)
    elif frac < 0.85:
        return 58.0 - (58.0 - 46.0) * ((frac - 0.15) / 0.70)
    else:
        return 46.0 - (46.0 - 25.0) * ((frac - 0.85) / 0.15)

def simulate_trajectory(args):
    seed, N, scaffold_dG_ref, duration = args
    np.random.seed(seed)
    
    state = [None] * N
    sim_time = 0.0
    steps = 0
    max_steps = 250000
    target_carries = [0] * N
    
    while sim_time < duration and steps < max_steps:
        T_c = get_temperature(sim_time, duration)
        T_k = T_c + 273.15
        
        scale = T_k / 338.15
        scaff_dG = scaffold_dG_ref * scale
        comp_dG = COMPUTE_MATCH_DG_REF * scale
        
        rates = []
        events = []
        
        for p in range(N):
            curr = state[p]
            left = state[p - 1] if p > 0 else None
            right = state[p + 1] if p < N - 1 else None
            candidates = [0] if p == 0 else [0, 1]
            
            if curr is None:
                for cand in candidates:
                    eff_conc = C_EFF_MOLAR if (p > 0 and left is not None and left == target_carries[p - 1]) else BULK_TILE_EXCESS
                    rates.append(K_ON * eff_conc)
                    events.append(('bind', p, cand))
            else:
                dG = scaff_dG[p]
                if p > 0 and left is not None:
                    dG += comp_dG if (left == target_carries[p - 1] and curr == target_carries[p]) else MISMATCH_PENALTY
                if p < N - 1 and right is not None:
                    dG += comp_dG if (right == target_carries[p + 1] and curr == target_carries[p]) else MISMATCH_PENALTY
                    
                k_off = K_ON * np.exp(min(40.0, dG / (R_KCAL * T_k)))
                rates.append(k_off)
                events.append(('unbind', p, None))
                
                for cand in candidates:
                    if cand != curr:
                        barrier = 0.5 if (p > 0 and left == target_carries[p - 1] and cand == target_carries[p]) else 2.8
                        k_rep = 2e4 * np.exp(-barrier / (R_KCAL * T_k))
                        rates.append(k_rep)
                        events.append(('replace', p, cand))
                        
        rates = np.array(rates)
        total_rate = np.sum(rates)
        if total_rate <= 0:
            break
            
        dt = -np.log(np.random.uniform(1e-12, 1.0)) / total_rate
        sim_time += dt
        steps += 1
        
        r = np.random.uniform(0.0, total_rate)
        idx = np.searchsorted(np.cumsum(rates), r)
        act, p, cand = events[idx]
        
        if act in ('bind', 'replace'):
            state[p] = cand
        elif act == 'unbind':
            state[p] = None
            
    errors = sum(1 for p in range(N) if state[p] != target_carries[p])
    return (errors == 0), errors

def run_production_benchmarks():
    with open("flattened_domains.json", "r") as f:
        domains = json.load(f)
        
    N = len(domains)
    scaff_dG_arr = np.array([domains[str(i)]["tuned_dG"] for i in range(N)])
    
    test_scales = [10, 15, 20, 25]
    anneal_durations = [300.0, 600.0, 900.0]
    
    print(f"{'Scaffold Length (N)':<20} | {'Anneal Time (s)':<16} | {'Yield (%)':<12} | {'Mean Errors':<12}", flush=True)
    print("-" * 65, flush=True)
    
    summary = {}
    for n in test_scales:
        for dur in anneal_durations:
            args_list = [(5000 + i, n, scaff_dG_arr[:n], dur) for i in range(20)]
            with Pool(processes=4) as pool:
                res = pool.map(simulate_trajectory, args_list)
                
            success = sum(1 for r in res if r[0])
            errs = [r[1] for r in res]
            yield_pct = (success / 20.0) * 100.0
            mean_err = float(np.mean(errs))
            
            key = f"N{n}_{int(dur)}s"
            summary[key] = {"yield": yield_pct, "errors": mean_err}
            print(f"N = {n:<16} | {dur:<16.1f} | {yield_pct:<12.1f} | {mean_err:<12.2f}", flush=True)
            
    with open("production_benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("\nScale-up production benchmark complete. Saved to production_benchmark_results.json", flush=True)

if __name__ == "__main__":
    run_production_benchmarks()
