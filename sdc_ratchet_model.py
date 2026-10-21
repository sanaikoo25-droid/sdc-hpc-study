import numpy as np
import json
import sys

R_KCAL = 0.0019872

def simulate_sdc_ratchet(N=25, M=3, anneal_duration=300.0, flattened=True, num_trials=20):
    """
    Faithful model of the experimental Scaffolded DNA Computer:
    - Position 0 is an Anchor (non-competitive seed)
    - Adjacent compute domains ratchet correct computation left-to-right
    - Branch migration allows tile replacement
    """
    k_on = 1e6
    tile_excess = 1.0e-6
    compute_match_dG_ref = -9.2   # 12-nt compute domain duplex free energy
    mismatch_penalty = +3.0       # algorithmic mismatch penalty

    np.random.seed(42)
    if flattened:
        scaffold_dG_ref = np.full(N, -12.5)  # Isoenergetic landscape
    else:
        scaffold_dG_ref = np.random.uniform(-20.9, -11.1, size=N)  # Raw M13

    target = [0] * N
    # Anchor at pos 0 is deterministic (1 tile), remaining positions have 2 competing tiles
    tile_pool = {0: [0]}
    for p in range(1, N):
        tile_pool[p] = [0, 1]

    completed_count = 0
    errors_list = []
    final_states = []

    for trial in range(num_trials):
        state = [None] * N
        sim_time = 0.0
        steps = 0
        max_steps = 300000

        while sim_time < anneal_duration and steps < max_steps:
            # Temperature annealing protocol: 80 C down to 25 C
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
                    # Association from solution
                    for cand in tile_pool[p]:
                        rates.append(k_on * tile_excess)
                        events.append(('bind', p, cand))
                else:
                    # Calculate unbinding free energy
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

                    # Bound tile dissociation
                    k_off = k_on * np.exp(min(40.0, dG / (R_KCAL * T_kelvin)))
                    rates.append(k_off)
                    events.append(('unbind', p, None))

                    # Competing tile exchange (branch migration replacement)
                    for cand in tile_pool[p]:
                        if cand != curr:
                            # If candidate matches the left neighbor's carry/state, replacement is facilitated
                            if p > 0 and left == target[p - 1] and cand == target[p]:
                                barrier = 0.5  # rapid toehold-mediated invasion
                            else:
                                barrier = 3.0  # high barrier for unfavorable swap
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
        errors_list.append(errors)
        if errors == 0:
            completed_count += 1
        final_states.append(state)

    yield_pct = (completed_count / num_trials) * 100.0
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
            y_val, err_val = simulate_sdc_ratchet(N=25, M=3, anneal_duration=t_dur, flattened=mode, num_trials=20)
            key = f"{label}_{int(t_dur)}s"
            results[key] = {"yield": y_val, "errors": err_val}
            print(f"{label:<16} | {t_dur:<16.1f} | {y_val:<10.1f} | {err_val:<12.2f}", flush=True)

    with open("ratchet_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nRatchet simulation finished and written to ratchet_results.json", flush=True)
