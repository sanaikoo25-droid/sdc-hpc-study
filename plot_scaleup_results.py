import json
import matplotlib
matplotlib.use('Agg')  # Force headless rendering without Qt/X11
import matplotlib.pyplot as plt
import numpy as np

with open("production_benchmark_results.json", "r") as f:
    data = json.load(f)

# Extract structured arrays
durations = [300, 600, 900]
lengths = [10, 15, 20, 25]

yield_matrix = {dur: [] for dur in durations}
error_matrix = {dur: [] for dur in durations}

for dur in durations:
    for n in lengths:
        key = f"N{n}_{dur}s"
        if key in data:
            yield_matrix[dur].append(data[key]["yield"])
            error_matrix[dur].append(data[key]["errors"])
        else:
            yield_matrix[dur].append(np.nan)
            error_matrix[dur].append(np.nan)

# Create 2-panel figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
colors = {300: "#1f77b4", 600: "#2ca02c", 900: "#d62728"}
markers = {300: "o", 600: "s", 900: "^"}

# Panel A: Ground-State Yield vs N
for dur in durations:
    ax1.plot(lengths, yield_matrix[dur], label=f"Anneal {dur} s", 
             color=colors[dur], marker=markers[dur], linewidth=2, markersize=8)

ax1.set_xlabel("Scaffold Length ($N$ positions)", fontsize=12, fontweight="bold")
ax1.set_ylabel("Ground-State Yield (%)", fontsize=12, fontweight="bold")
ax1.set_title("A: Computational Yield vs System Scale", fontsize=13, fontweight="bold")
ax1.set_xticks(lengths)
ax1.set_ylim(-2, 45)
ax1.grid(True, linestyle="--", alpha=0.6)
ax1.legend(frameon=True, fontsize=10)

# Panel B: Mean Bit Errors vs N
for dur in durations:
    ax2.plot(lengths, error_matrix[dur], label=f"Anneal {dur} s", 
             color=colors[dur], marker=markers[dur], linewidth=2, markersize=8)

ax2.set_xlabel("Scaffold Length ($N$ positions)", fontsize=12, fontweight="bold")
ax2.set_ylabel("Mean Bit Errors (out of $N$)", fontsize=12, fontweight="bold")
ax2.set_title("B: Error Scaling Across Array Size", fontsize=13, fontweight="bold")
ax2.set_xticks(lengths)
ax2.set_ylim(0, 3.0)
ax2.grid(True, linestyle="--", alpha=0.6)
ax2.legend(frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig("sdc_scaleup_performance.png", dpi=300)
print("Saved publication plot: sdc_scaleup_performance.png")
