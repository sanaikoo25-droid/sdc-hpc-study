     # Multi-Scale HPC Study of Scaffolded DNA Computers (SDC)

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![oxDNA2](https://img.shields.io/badge/simulation-oxDNA2-brightgreen.svg)](https://dna.physics.ox.ac.uk/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)

This repository provides the multi-scale computational framework, simulation codes, coarse-grained models, and benchmark datasets for:

> **"Microscopic Free-Energy Landscaping and Coarse-Grained 3D Dynamics of Scaffolded DNA Computers: Overcoming Kinetic Traps at Scale"**

---

## Scientific Background and Key Findings

Thermodynamically favoured molecular computing embeds algorithmic outputs into thermodynamic ground states (minimum free energy configurations), bypassing energy-consuming kinetic proofreading. However, physical implementations using biologically sourced scaffolds (e.g., M13) suffer from wide sequence-dependent duplex free-energy variance (Delta G ranging from -20.9 to -11.1 kcal/mol), causing deep kinetic trapping along long non-sink logic runs.

Through this multi-scale HPC investigation, we resolved four foundational challenges:

1. **Solution-Phase Toehold Displacement:** 3D coarse-grained simulations in oxDNA2 demonstrated that adjacent bound compute domains are separated by an average physical distance of 18.43 nm, yielding a 0.0% direct contact probability. This proves that error correction does not operate via on-scaffold cross-hybridization, but rather via solution-phase tile displacement.
2. **Quantification of Effective Local Concentration:** By calculating the radial volume sampled by the flexible single-stranded 12-nt compute arms (RMSF = 10.14 nm), we quantified an effective tethered concentration of C_eff = 63.37 microM. This imparts a 63-fold kinetic acceleration for matching incoming tiles over bulk solution (1.0 microM), powering the directional assembly ratchet without enzyme machinery.
3. **Hierarchical Energy Landscaping:** We resolved why naive sequence flattening to -12.0 kcal/mol failed (0.0% yield due to global tile detachment). By discovering a hierarchical optimization threshold (Delta G_scaffold = -18.5 kcal/mol), biological scaffold variance was homogenized (Delta G = -18.5 +- 0.6 kcal/mol) while preserving scaffold occupancy and a Delta-Delta-G = 12.7 kcal/mol proofreading gap.
4. **Sub-Linear Error Scaling up to N = 50:** By coupling an intermediate selective-dwell thermal annealing protocol (58 C to 46 C) with anchor-driven seeding, computational error accumulation scales sub-linearly (<= 2.0 bit errors) up to 200 bits of computation (N = 50).

---

## Repository File Structure

* `generate_sdc_oxdna.py` - Generates 3D oxDNA2 topology and configuration files
* `oxdna_relax.txt` - oxDNA steepest-descent energy minimization setup
* `oxdna_sdc_input.txt` - oxDNA Brownian dynamics production input
* `analyze_sdc_structure.py` - Structural trajectory parser (RMSF, extension, distances)
* `calc_ceff.py` - Effective local concentration (C_eff) calculation script
* `compute_contact_map.py` - Minimum inter-arm distance and contact probability calculator
* `oxdna_to_pdb_fixed.py` - oxDNA-to-PDB coordinate converter
* `unwrap_perfect_sdc.py` - Periodic boundary condition unwrapper for assembled complexes
* `scaffold_energy_flattener.py` - MCMC sequence optimizer for biological scaffold homogenization
* `sdc_compiler.py` - FSM-to-SDC tile library compiler
* `sdc_kmc.py` - Single-trajectory continuous-time Gillespie KMC simulator
* `sdc_m_scaling.py` - Kinetic trapping analyzer for non-sink carry runs
* `sdc_ratchet_parallel.py` - Multi-core directional ratchet simulator
* `sdc_production_hpc.py` - 3D-calibrated scale-up benchmark simulator (N=10 to N=25)
* `sdc_scaleup_n50.py` - Large-scale benchmark simulator for N=30 to N=50
* `plot_scaleup_results.py` - Generates Figure 2 (Scaling Yield and Errors vs. N)
* `plot_energy_landscape.py` - Generates Figure 3 (Hierarchical Energy Level Diagram)
* `submit_oxdna.pbs` - PBS batch script for oxDNA MD runs
* `submit_sdc.pbs` - PBS batch script for parallel KMC execution
* `sdc_4pos.top` - oxDNA topology definition (288 nucleotides, 5 strands)
* `sdc_4pos.dat` - Initial coordinates for the 4-position complex
* `sdc_lastconf.dat` - Equilibrated coordinates after Brownian MD
* `sdc_final_structure.pdb` - Assembled 3D coordinate file for PyMOL / oxView
* `flattened_domains.json` - Mismatch-engineered sequences and binding free energies
* `compiled_addition_fsm.json` - Compiled 25-bit ADDITION truth tables and tile definitions
* `production_benchmark_results.json` - Raw KMC benchmark statistics (N=10 to N=25)
* `n50_scaleup_results.json` - Raw KMC benchmark statistics (N=30 to N=50)
* `sdc_scaleup_performance.png` - Figure 2: Yield and Mean Errors vs N
* `sdc_energy_landscape.png` - Figure 3: Energy separation and scaffold homogenization

---

## Installation and Prerequisites

To execute the simulation pipelines, set up a Conda environment containing Python 3.10 and the scientific computing stack:

```bash
conda create -n sdc_sim python=3.10 -y
conda activate sdc_sim
conda install -y numpy scipy matplotlib pandas h5py
pip install tqdm

## Summary Benchmark Performance

| Scaffold Size (N) | Total Compute Bits | Anneal Duration (s) | Ground-State Yield (%) | Mean Bit Errors | Computational Regime |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **N = 10** | 40 | 600 | **40.0%** | **0.70** | Fast, reversible kinetics |
| **N = 15** | 60 | 600 | **30.0%** | **1.05** | Directional ratchet |
| **N = 20** | 80 | 600 | **20.0%** | **1.20** | Intermediate scale |
| **N = 25** | 100 | 600 | **25.0%** | **1.50** | Resonant proofreading dwell |
| **N = 30** | 120 | 600 | **15.0%** | **1.55** | Sub-linear error regime |
| **N = 40** | 160 | 600 | **15.0%** | **2.00** | Stable plateau |
| **N = 50** | 200 | 600 | **10.0%** | **2.00** | 200-bit computation |

Citation and Attribution

If you utilize the code, data, or structural models in this repository, please cite:

@misc{Naikoo2026SDC,
  author       = {Sharif Ahmad Naikoo},
  title        = {Microscopic Free-Energy Landscaping and Coarse-Grained 3D Dynamics of Scaffolded DNA Computers: Overcoming Kinetic Traps at Scale},
  year         = {2026},
  institution  = {Department of Chemistry, Indian Institute of Technology Kanpur},
  howpublished = {[https://github.com/sanaikoo25-droid/sdc-hpc-study](https://github.com/sanaikoo25-droid/sdc-hpc-study)},
  note         = {Computational Biophysics Research Preprint}
}


License

This project is licensed under the MIT License.

