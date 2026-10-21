import numpy as np
import json
import sys

R_KCAL = 0.0019872

def run_addition_m_experiment(N=25, M=5, num_trials=10, T_celsius=55.0, max_time=300.0):
    T_kelvin = T_celsius + 273.15
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG = -8.5
    mismatch_penalty = +4.5

    np.random.seed(101)
    scaffold_dG = np.random.uniform(-20.9, -11.1, size=N)

    target = [0] * N
    tile_pool = {p: [0, 1] for p in range(N)}

    completion_times = []
    successes = 0

    for trial in range(num_trials):
        state = [None] * N
        sim_time = 0.0
        steps = 0
        max_steps = 60000

        while steps < max_steps and sim_time < max_time:
            if state == target:
                successes += 1
                completion_times.append(sim_time)
                break

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
                    if left is not None and left == target[p - 1] and curr == target[p]:
                        dG += compute_match_dG
                    elif left is not None:
                        dG += mismatch_penalty

                    if right is not None and right == target[p + 1] and curr == target[p]:
                        dG += compute_match_dG
                    elif right is not None:
                        dG += mismatch_penalty

                    k_off = k_on * np.exp(dG / (R_KCAL * T_kelvin))
                    rates.append(k_off)
                    events.append(('unbind', p, None))

                    for cand in tile_pool[p]:
                        if cand != curr:
                            barrier = 2.0 if (N - M <= p < N) else 0.5
                            k_replace = 1e3 * np.exp(-barrier / (R_KCAL * T_kelvin))
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

    yield_pct = (successes / num_trials) * 100.0
    mean_time = float(np.mean(completion_times)) if completion_times else float('nan')
    return yield_pct, mean_time

if __name__ == "__main__":
    m_values = [1, 2, 3, 4, 5]
    print(f"{'M (Non-Sink)':<15} | {'Yield (%)':<12} | {'MFPT (s)':<15}", flush=True)
    print("-" * 50, flush=True)

    results = {}
    for m in m_values:
        y_rate, mfpt = run_addition_m_experiment(N=25, M=m, num_trials=10, T_celsius=55.0)
        results[str(m)] = {"yield": y_rate, "mfpt": mfpt}
        print(f"{m:<15} | {y_rate:<12.1f} | {mfpt:<15.3f}", flush=True)

    with open("m_scaling_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("All results successfully saved to m_scaling_results.json", flush=True)
