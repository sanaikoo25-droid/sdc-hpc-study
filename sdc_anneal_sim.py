import numpy as np
import json
import sys

R_KCAL = 0.0019872

def simulate_sdc_anneal(N=25, M=3, anneal_duration=60.0, num_trials=10):
    """
    Simulates SDC computation during an active temperature annealing ramp:
    T drops linearly from 80 C to 25 C over anneal_duration (seconds).
    N: total scaffold positions
    M: length of consecutive non-sink carry run
    """
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG_ref = -8.5  # kcal/mol at reference
    mismatch_penalty = +3.5      # enthalpic penalty for mismatched compute domain

    # Heterogeneous scaffold binding energies (M13 scaffold from paper)
    np.random.seed(42)
    scaffold_dG_ref = np.random.uniform(-20.9, -11.1, size=N)

    target = [0] * N
    tile_pool = {p: [0, 1] for p in range(N)}

    completed_count = 0
    final_errors_per_trial = []

    for trial in range(num_trials):
        state = [None] * N
        sim_time = 0.0
        steps = 0
        max_steps = 100000

        while sim_time < anneal_duration and steps < max_steps:
            # Linear temperature ramp: 80 C down to 25 C
            progress = min(1.0, sim_time / anneal_duration)
            T_celsius = 80.0 - (80.0 - 25.0) * progress
            T_kelvin = T_celsius + 273.15

            # Scaled free energies at current temperature
            scale_factor = T_kelvin / 338.15
            scaffold_dG = scaffold_dG_ref * scale_factor
            compute_match_dG = compute_match_dG_ref * scale_factor

            rates = []
            events = []

            for p in range(N):
                curr = state[p]
                left = state[p - 1] if p > 0 else None
                right = state[p + 1] if p < N - 1 else None

                if curr is None:
                    # Association
                    for cand in tile_pool[p]:
                        rates.append(k_on * tile_excess)
                        events.append(('bind', p, cand))
                else:
                    # Dissociation
                    dG = scaffold_dG[p]
                    if left is not None and left == target[p - 1] and curr == target[p]:
                        dG += compute_match_dG
                    elif left is not None and left != curr:
                        dG += mismatch_penalty

                    if right is not None and right == target[p + 1] and curr == target[p]:
                        dG += compute_match_dG
                    elif right is not None and right != curr:
                        dG += mismatch_penalty

                    k_off = k_on * np.exp(min(40.0, dG / (R_KCAL * T_kelvin)))
                    rates.append(k_off)
                    events.append(('unbind', p, None))

                    # Competitive displacement
                    for cand in tile_pool[p]:
                        if cand != curr:
                            barrier = 2.0 if (N - M <= p < N) else 0.5
                            k_replace = 5e3 * np.exp(-barrier / (R_KCAL * T_kelvin))
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

        # Tally final configuration state at end of anneal
        errors = sum(1 for p in range(N) if state[p] != target[p])
        final_errors_per_trial.append(errors)
        if errors == 0:
            completed_count += 1

    yield_pct = (completed_count / num_trials) * 100.0
    mean_errors = float(np.mean(final_errors_per_trial))
    return yield_pct, mean_errors

if __name__ == "__main__":
    print(f"{'M (Non-Sink)':<15} | {'Yield (%)':<12} | {'Mean Bit Errors (out of 25)':<28}", flush=True)
    print("-" * 60, flush=True)

    results = {}
    for m in [1, 2, 3, 4, 5]:
        yield_rate, avg_err = simulate_sdc_anneal(N=25, M=m, anneal_duration=30.0, num_trials=10)
        results[str(m)] = {"yield": yield_rate, "mean_errors": avg_err}
        print(f"{m:<15} | {yield_rate:<12.1f} | {avg_err:<28.2f}", flush=True)

    with open("anneal_m_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nAnneal simulation complete. Saved to anneal_m_results.json", flush=True)
