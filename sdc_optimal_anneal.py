import numpy as np
import json
import sys
from multiprocessing import Pool

R_KCAL = 0.0019872

def get_temperature(sim_time, total_duration, protocol="selective"):
    """
    linear: uniform dT/dt from 80 C to 25 C
    selective: rapid drop 80 -> 58 C (15% time), long dwell 58 -> 46 C (70% time), finish 46 -> 25 C (15% time)
    """
    frac = min(1.0, sim_time / total_duration)
    if protocol == "linear":
        return 80.0 - (80.0 - 25.0) * frac
    elif protocol == "selective":
        if frac < 0.15:
            # Fast drop through unbound regime
            sub = frac / 0.15
            return 80.0 - (80.0 - 58.0) * sub
        elif frac < 0.85:
            # Long dwell inside error-correction proofreading window
            sub = (frac - 0.15) / 0.70
            return 58.0 - (58.0 - 46.0) * sub
        else:
            # Cool down to lock ground state
            sub = (frac - 0.85) / 0.15
            return 46.0 - (46.0 - 25.0) * sub

def worker_trajectory(args):
    seed, N, M, duration, protocol = args
    np.random.seed(seed)
    
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG_ref = -9.2
    mismatch_penalty = +3.5
    
    # Raw M13 biological scaffold energetics
    scaffold_rng = np.random.RandomState(42)
    scaffold_dG_ref = scaffold_rng.uniform(-20.9, -11.1, size=N)
    
    target = [0] * N
    tile_pool = {0: [0]}  # Deterministic anchor
    for p in range(1, N):
        tile_pool[p] = [0, 1]
        
    state = [None] * N
    sim_time = 0.0
    steps = 0
    max_steps = 250000
    
    while sim_time < duration and steps < max_steps:
        T_celsius = get_temperature(sim_time, duration, protocol)
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
                        # Ratchet replacement: favored if left neighbor is correct
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

def run_suite(N=25, M=3, duration=600.0, protocol="selective", num_trials=20, n_cpus=4):
    args_list = [(2000 + i, N, M, duration, protocol) for i in range(num_trials)]
    with Pool(processes=n_cpus) as pool:
        results = pool.map(worker_trajectory, args_list)
        
    successes = sum(1 for r in results if r[0])
    errors = [r[1] for r in results]
    yield_pct = (successes / num_trials) * 100.0
    mean_err = float(np.mean(errors))
    return yield_pct, mean_err

if __name__ == "__main__":
    durations = [300.0, 600.0, 1200.0]
    print(f"{'Protocol':<18} | {'Duration (s)':<14} | {'Yield (%)':<10} | {'Mean Errors':<12}", flush=True)
    print("-" * 60, flush=True)
    
    results = {}
    for proto in ["linear", "selective"]:
        for d in durations:
            y, e = run_suite(N=25, M=3, duration=d, protocol=proto, num_trials=20, n_cpus=4)
            key = f"{proto}_{int(d)}s"
            results[key] = {"yield": y, "mean_errors": e}
            print(f"{proto:<18} | {d:<14.1f} | {y:<10.1f} | {e:<12.2f}", flush=True)
            
    with open("optimal_anneal_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAnneal protocol optimization run complete. Saved to optimal_anneal_results.json", flush=True)
