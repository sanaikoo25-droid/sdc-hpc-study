# Multi-Scale HPC Study of Scaffolded DNA Computers (SDC)

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![oxDNA2](https://img.shields.io/badge/simulation-oxDNA2-brightgreen.svg)](https://dna.physics.ox.ac.uk/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

This repository provides the multi-scale computational framework, simulation codes, coarse-grained models, and benchmark datasets for:

> **"Microscopic Free-Energy Landscaping and Coarse-Grained 3D Dynamics of Scaffolded DNA Computers: Overcoming Kinetic Traps at Scale"**

---

## 📌 Scientific Background & Key Findings

Thermodynamically favoured molecular computing embeds algorithmic outputs into thermodynamic ground states (minimum free energy configurations), bypassing energy-consuming kinetic proofreading. However, physical implementations using biologically sourced scaffolds (e.g., M13) suffer from wide sequence-dependent duplex free-energy variance ($\Delta G^\circ$ ranging from $-20.9$ to $-11.1\text{ kcal/mol}$), causing deep kinetic trapping along long non-sink logic runs.

Through this multi-scale HPC investigation, we resolved four foundational challenges:

1. **Solution-Phase Toehold Displacement:** 3D coarse-grained simulations in **oxDNA2** demonstrated that adjacent bound compute domains are separated by an average physical distance of **$18.43 \pm 5.58\text{ nm}$**, yielding a **$0.0\%$ direct contact probability**. This proves that error correction does *not* operate via on-scaffold cross-hybridization, but rather via solution-phase tile displacement.
2. **Quantification of Effective Local Concentration ($C_{\text{eff}}$):** By calculating the radial volume sampled by the flexible single-stranded 12-nt compute arms ($\text{RMSF} = 10.14\text{ nm}$), we quantified an effective tethered concentration of **$C_{\text{eff}} \approx 63.37\ \mu\text{M}$**. This imparts a **63-fold kinetic acceleration** for matching incoming tiles over bulk solution ($1.0\ \mu\text{M}$), powering the directional assembly ratchet without enzyme machinery.
3. **Hierarchical Energy Landscaping:** We resolved why naive sequence flattening to $-12.0\text{ kcal/mol}$ failed ($0.0\%$ yield due to global tile detachment). By discovering a hierarchical optimization threshold ($\Delta G^\circ_{\text{scaffold}} \approx -18.5\text{ kcal/mol}$), biological scaffold variance was homogenized ($\Delta G^\circ \approx -18.5 \pm 0.6\text{ kcal/mol}$) while preserving scaffold occupancy and a $\Delta\Delta G = 12.7\text{ kcal/mol}$ proofreading gap.
4. **Sub-Linear Error Scaling up to $N = 50$:** By coupling an intermediate selective-dwell thermal annealing protocol ($58^\circ\text{C} \to 46^\circ\text{C}$) with anchor-driven seeding, computational error accumulation scales sub-linearly ($\le 2.0$ bit errors) up to 200 bits of computation ($N = 50$).

---

## 📂 Repository File Structure       
