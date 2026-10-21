import numpy as np
import json
import time

R_KCAL = 0.0019872  # kcal / (mol * K)

class SDCKineticModel:
    def __init__(self, N, target_states, tile_pool, scaffold_dG, compute_match_dG=-9.0, k_on=1e6):
        self.N = N
        self.target_states = target_states
        self.tile_pool = tile_pool
        self.scaffold_dG = np.array(scaffold_dG)
        self.compute_match_dG = compute_match_dG
        self.k_on = k_on
        self.tile_excess_conc = 1.0e-6  # 1 uM (excess over scaffold)

    def calculate_tile_dG(self, pos, tile_id, left_tile, right_tile, T_kelvin):
        dG = self.scaffold_dG[pos]
        if pos > 0 and left_tile is not None:
            if left_tile == self.target_states[pos - 1] and tile_id == self.target_states[pos]:
                dG += self.compute_match_dG
        if pos < self.N - 1 and right_tile is not None:
            if right_tile == self.target_states[pos + 1] and tile_id == self.target_states[pos]:
                dG += self.compute_match_dG
        return dG

    def run_trajectory(self, T_celsius, max_time=3600.0, max_steps=500000):
        T_kelvin = T_celsius + 273.15
        state = [None] * self.N
        sim_time = 0.0

        for step in range(max_steps):
            if state == self.target_states:
                return sim_time, True, step

            rates = []
            events = []

            for p in range(self.N):
                current = state[p]
                left = state[p - 1] if p > 0 else None
                right = state[p + 1] if p < self.N - 1 else None

                if current is None:
                    for candidate in self.tile_pool[p]:
                        rates.append(self.k_on * self.tile_excess_conc)
                        events.append(('bind', p, candidate))
                else:
                    dG = self.calculate_tile_dG(p, current, left, right, T_kelvin)
                    k_off = self.k_on * np.exp(dG / (R_KCAL * T_kelvin))
                    rates.append(k_off)
                    events.append(('unbind', p, None))

                    for candidate in self.tile_pool[p]:
                        if candidate != current:
                            dG_cand = self.calculate_tile_dG(p, candidate, left, right, T_kelvin)
                            delta = dG_cand - dG
                            k_replace = 1e3 * min(1.0, np.exp(-delta / (R_KCAL * T_kelvin)))
                            rates.append(k_replace)
                            events.append(('replace', p, candidate))

            rates = np.array(rates)
            total_rate = np.sum(rates)
            if total_rate <= 0:
                break

            dt = -np.log(np.random.uniform(1e-12, 1.0)) / total_rate
            sim_time += dt
            if sim_time > max_time:
                break

            r = np.random.uniform(0.0, total_rate)
            idx = np.searchsorted(np.cumsum(rates), r)
            action, p, candidate = events[idx]

            if action in ('bind', 'replace'):
                state[p] = candidate
            elif action == 'unbind':
                state[p] = None

        return sim_time, (state == self.target_states), max_steps

if __name__ == "__main__":
    N = 25
    np.random.seed(42)
    # Variance of M13 scaffold dG (-20.9 to -11.1 kcal/mol)
    scaffold_dG_m13 = np.random.uniform(-20.9, -11.1, size=N)
    target = [0] * N
    tile_pool = {i: [0, 1] for i in range(N)}

    model = SDCKineticModel(N=N, target_states=target, tile_pool=tile_pool, scaffold_dG=scaffold_dG_m13)

    print(f"--- Running Benchmark KMC on N={N} Scaffolded DNA Computer ---")
    start = time.time()
    for temp in [50.0, 55.0, 60.0]:
        t_sim, success, steps = model.run_trajectory(T_celsius=temp, max_time=120.0, max_steps=100000)
        status = "COMPLETED" if success else "TIMEOUT/TRAPPED"
        print(f"T = {temp} C | Status: {status} | Sim Time: {t_sim:.3f} s | Steps: {steps}")
    print(f"Wall time: {time.time() - start:.2f} s")
