import numpy as np
import json
import sys
from multiprocessing import Pool

R_KCAL = 0.0019872

def single_trajectory_worker(args):
    """Executes a single stochastic anneal trajectory."""
    seed, N, M, anneal_duration, flattened = args
    np.random.seed(seed)
    
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG_ref = -9.2
    mismatch_penalty = +3.0
    
    if flattened:
        scaffold_dG_ref = np.full(N, -12.5)
    else:
        # Recreate deterministic M13 scaffold per paper
        scaffold_rng = np.random.RandomState(42)
        scaffold_dG_ref = scaffold_rng.uniform(-20.9, -11.1, size=N)
        
    target = [0] * N
    tile_pool = {0: [0]}
    for p in range(1, N):
        tile_pool[p] = [0, 1]
        
    state = [None] * N
    sim_time = 0.0
    steps = 0
    max_steps = 150000
    
    while sim_time < anneal_duration and steps < max_steps:
        progress = min(1.0, sim_time / anneal_duration)
        T_celsius = 80.0 - (80.0 - 25.0) * progress
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
                            barrier = 3.0
                        k_replace = 1e4 * np.exp(-barrier / (R_KCAL * T_kelvin))
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

def run_experiment(N=25, M=3, anneal_duration=300.0, flattened=True, num_trials=20, n_cpus=4):
    args_list = [(1000 + i, N, M, anneal_duration, flattened) for i in range(num_trials)]
    with Pool(processes=n_cpus) as pool:
        results = pool.map(single_trajectory_worker, args_list)
        
    successes = sum(1 for res in results if res[0])
    errors_list = [res[1] for res in results]
    yield_pct = (successes / num_trials) * 100.0
    mean_err = float(np.mean(errors_list))
    return yield_pct, mean_err

if __name__ == "__main__":
    durations = [120.0, 300.0, 600.0]
    print(f"{'Landscape':<16} | {'Anneal Time (s)':<16} | {'Yield (%)':<10} | {'Mean Errors':<12}", flush=True)
    print("-" * 62, flush=True)

    results = {}
    for mode in [False, True]:
        label = "Flattened" if mode else "Raw M13"
        for t_dur in durations:
            y_val, err_val = run_experiment(N=25, M=3, anneal_duration=t_dur, flattened=mode, num_trials=20, n_cpus=4)
            key = f"{label}_{int(t_dur)}s"
            results[key] = {"yield": y_val, "errors": err_val}
            print(f"{label:<16} | {t_dur:<16.1f} | {y_val:<10.1f} | {err_val:<12.2f}", flush=True)

    with open("ratchet_parallel_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nParallel simulation complete and saved to ratchet_parallel_results.json", flush=True)
