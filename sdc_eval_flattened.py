import numpy as np
import json
import sys
from multiprocessing import Pool

R_KCAL = 0.0019872

def get_temperature(sim_time, total_duration):
    """Selective temperature annealing profile: dwells inside 58 C -> 46 C."""
    frac = min(1.0, sim_time / total_duration)
    if frac < 0.15:
        return 80.0 - (80.0 - 58.0) * (frac / 0.15)
    elif frac < 0.85:
        return 58.0 - (58.0 - 46.0) * ((frac - 0.15) / 0.70)
    else:
        return 46.0 - (46.0 - 25.0) * ((frac - 0.85) / 0.15)

def worker_trajectory(args):
    seed, N, scaffold_dG_ref, duration = args
    np.random.seed(seed)
    
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG_ref = -9.2
    mismatch_penalty = +3.5
    
    target = [0] * N
    tile_pool = {0: [0]}  # Deterministic anchor
    for p in range(1, N):
        tile_pool[p] = [0, 1]
        
    state = [None] * N
    sim_time = 0.0
    steps = 0
    max_steps = 250000
    
    while sim_time < duration and steps < max_steps:
        T_celsius = get_temperature(sim_time, duration)
        T_kelvin = T_celsius + 273.15
        
        scale = T_kelvin / 338.15
        scaffold_dG = scaffold_dG_ref * scale
        compute_match_dG = compute_match_dG_ref * scale
        
        rates = []
        events = []
        
        for p in range(N):
            curr = state[p]
            left = state[p - 1] if p > 0 else None
            right = state[p + 1] if p < N - 1 else None
            
            if curr is None:
                for cand in tile_pool[p]:
                    rates.append(k_on * tile_excess)
                    events.append(('bind', p, cand))
            else:
                dG = scaffold_dG[p]
                if p > 0 and left is not None:
                    if left == target[p - 1] and curr == target[p]:
                        dG += compute_match_dG
                    else:
                        dG += mismatch_penalty
                if p < N - 1 and right is not None:
                    if right == target[p + 1] and curr == target[p]:
                        dG += compute_match_dG
                    else:
                        dG += mismatch_penalty
                        
                k_off = k_on * np.exp(min(40.0, dG / (R_KCAL * T_kelvin)))
                rates.append(k_off)
                events.append(('unbind', p, None))
                
                for cand in tile_pool[p]:
                    if cand != curr:
                        if p > 0 and left == target[p - 1] and cand == target[p]:
                            barrier = 0.5
                        else:
                            barrier = 2.8
                        k_replace = 2e4 * np.exp(-barrier / (R_KCAL * T_kelvin))
                        rates.append(k_replace)
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
        action, p, cand = events[idx]
        
        if action in ('bind', 'replace'):
            state[p] = cand
        elif action == 'unbind':
            state[p] = None
            
    errors = sum(1 for p in range(N) if state[p] != target[p])
    return (errors == 0), errors

def run_evaluation(num_trials=20, n_cpus=4):
    with open("flattened_domains.json", "r") as f:
        domains_data = json.load(f)
        
    N = len(domains_data)
    orig_dG_arr = np.array([domains_data[str(i)]["original_dG"] for i in range(N)])
    tuned_dG_arr = np.array([domains_data[str(i)]["tuned_dG"] for i in range(N)])
    
    durations = [300.0, 600.0, 1200.0]
    
    print(f"{'Landscape Type':<22} | {'Duration (s)':<14} | {'Yield (%)':<10} | {'Mean Errors':<12}", flush=True)
    print("-" * 65, flush=True)
    
    eval_summary = {}
    
    for label, dG_arr in [("Raw (Unflattened)", orig_dG_arr), ("Optimized (Flattened)", tuned_dG_arr)]:
        for d in durations:
            args_list = [(3000 + i, N, dG_arr, d) for i in range(num_trials)]
            with Pool(processes=n_cpus) as pool:
                res = pool.map(worker_trajectory, args_list)
                
            successes = sum(1 for r in res if r[0])
            err_vals = [r[1] for r in res]
            yield_pct = (successes / num_trials) * 100.0
            mean_err = float(np.mean(err_vals))
            
            key = f"{label}_{int(d)}s"
            eval_summary[key] = {"yield": yield_pct, "mean_errors": mean_err}
            print(f"{label:<22} | {d:<14.1f} | {yield_pct:<10.1f} | {mean_err:<12.2f}", flush=True)
            
    with open("flattened_comparison_results.json", "w") as f:
        json.dump(eval_summary, f, indent=4)
    print("\nComparison successfully finished and saved to flattened_comparison_results.json", flush=True)

if __name__ == "__main__":
    run_evaluation()
