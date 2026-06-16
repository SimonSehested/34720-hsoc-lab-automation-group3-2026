# Repo Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganiser det flade repository til en struktureret mappeopdeling: `src/`, `scripts/`, `tests/`, `data/`, `output/figures/`, `output/logs/`.

**Architecture:** Alle ændringer er fil-flytninger via `git mv` (bevarer git-historik). `conftest.py` opdateres *før* filflytning, så Python-imports virker under og efter oprydningen. `__pycache__`-mapper fjernes fra git tracking.

**Tech Stack:** Git Bash, pytest

---

### Task 1: Opdater conftest.py — tilføj src/ til sys.path

**Files:**
- Modify: `conftest.py`

*Skal ske FØR nogen filer flyttes — ellers fejler imports i tests, når kildekode rykkes til `src/`.*

- [ ] **Step 1: Erstat indholdet af conftest.py**

```python
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
```

- [ ] **Step 2: Bekræft at tests stadig kører (baseline)**

```bash
pytest tests/test_optimize_c2f2.py -v
```

Forventet: alle tests PASS (filer er endnu ikke flyttet, men sys.path er klar til det).

- [ ] **Step 3: Commit**

```bash
git add conftest.py
git commit -m "chore: tilføj src/ til sys.path i conftest"
```

---

### Task 2: Flyt kildekode til src/

**Files:**
- Move: `optimize_c2f2.py` → `src/optimize_c2f2.py`
- Move: `laser_polarization_optimizer.py` → `src/laser_polarization_optimizer.py`
- Move: `monitor.py` → `src/monitor.py`
- Move: `pol_cons.py` → `src/pol_cons.py`
- Move: `poli2.ipynb` → `src/poli2.ipynb`

- [ ] **Step 1: Opret src/ og flyt filer**

```bash
mkdir -p src
git mv optimize_c2f2.py src/
git mv laser_polarization_optimizer.py src/
git mv monitor.py src/
git mv pol_cons.py src/
git mv poli2.ipynb src/
```

- [ ] **Step 2: Bekræft imports virker efter flytning**

```bash
pytest tests/test_optimize_c2f2.py -v
```

Forventet: alle tests PASS — conftest.py tilføjer `src/` til sys.path, så `import optimize_c2f2` finder `src/optimize_c2f2.py`.

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: flyt kildekode til src/"
```

---

### Task 3: Flyt plotscripts til scripts/

**Files:**
- Move: `generate_100_graphs.py` → `scripts/`
- Move: `generate_new_measurement_graphs.py` → `scripts/`
- Move: `plot_benchmark.py` → `scripts/`
- Move: `plot_calibration_time.py` → `scripts/`
- Move: `plot_drift_analysis.py` → `scripts/`
- Move: `plot_monitor.py` → `scripts/`
- Move: `plot_threshold.py` → `scripts/`
- Move: `report_plots.py` → `scripts/`

- [ ] **Step 1: Opret scripts/ og flyt filer**

```bash
mkdir -p scripts
git mv generate_100_graphs.py scripts/
git mv generate_new_measurement_graphs.py scripts/
git mv plot_benchmark.py scripts/
git mv plot_calibration_time.py scripts/
git mv plot_drift_analysis.py scripts/
git mv plot_monitor.py scripts/
git mv plot_threshold.py scripts/
git mv report_plots.py scripts/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: flyt plot- og genererings-scripts til scripts/"
```

---

### Task 4: Flyt testhjælpescripts til tests/

**Files:**
- Move: `test_loop.py` → `tests/test_loop.py`
- Move: `run_continuous_test.py` → `tests/run_continuous_test.py`

- [ ] **Step 1: Flyt filer**

```bash
git mv test_loop.py tests/
git mv run_continuous_test.py tests/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: flyt testscripts til tests/"
```

---

### Task 5: Flyt CSV-datafiler til data/

**Files:**
- Move: `benchmark_raw.csv` → `data/`
- Move: `benchmark_summary.csv` → `data/`
- Move: `benchmark_summary_c1_c3.csv` → `data/`
- Move: `drift.csv` → `data/`
- Move: `monitor_log.csv` → `data/`
- Move: `opt_results.csv` → `data/`

- [ ] **Step 1: Opret data/ og flyt filer**

```bash
mkdir -p data
git mv benchmark_raw.csv data/
git mv benchmark_summary.csv data/
git mv benchmark_summary_c1_c3.csv data/
git mv drift.csv data/
git mv monitor_log.csv data/
git mv opt_results.csv data/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: flyt datafiler til data/"
```

---

### Task 6: Flyt billeder og grafer til output/figures/

**Files:**
- Move: `benchmark_*.png` (8 filer) → `output/figures/`
- Move: `report_fig*.png` (14 filer) → `output/figures/`
- Move: `IMG_6423.HEIC` → `output/figures/`
- Move: `graphs_100/` → `output/figures/graphs_100/`
- Move: `graphs_new_measurements/` → `output/figures/graphs_new_measurements/`
- Move: `plots/` → `output/figures/plots/`

- [ ] **Step 1: Opret output/figures/**

```bash
mkdir -p output/figures
```

- [ ] **Step 2: Flyt rod-PNG'er og HEIC**

```bash
git mv benchmark_drift.png output/figures/
git mv benchmark_drift2.png output/figures/
git mv benchmark_heatmaps.png output/figures/
git mv benchmark_heatmaps2.png output/figures/
git mv benchmark_pareto.png output/figures/
git mv benchmark_pareto2.png output/figures/
git mv benchmark_trials.png output/figures/
git mv benchmark_trials2.png output/figures/
git mv report_fig1_heatmaps.png output/figures/
git mv report_fig1_heatmaps_c1_c3.png output/figures/
git mv report_fig10_movement_vs_power_c1_c3.png output/figures/
git mv report_fig11_best_config_arm_positions_c1_c3.png output/figures/
git mv report_fig2_lineplots.png output/figures/
git mv report_fig2_power_lines_c1_c3.png output/figures/
git mv report_fig3_pareto.png output/figures/
git mv report_fig3_runtime_lines_c1_c3.png output/figures/
git mv report_fig4_boxplots.png output/figures/
git mv report_fig4_pareto_c1_c3.png output/figures/
git mv report_fig5_boxplots_all_c1_c3.png output/figures/
git mv report_fig5_ranking.png output/figures/
git mv report_fig6_success_rates_c1_c3.png output/figures/
git mv report_fig7_ranking_c1_c3.png output/figures/
git mv report_fig8_balanced_ranking_c1_c3.png output/figures/
git mv report_fig9_measurement_order_c1_c3.png output/figures/
git mv IMG_6423.HEIC output/figures/
```

- [ ] **Step 3: Flyt grafdirektories**

```bash
git mv graphs_100 output/figures/graphs_100
git mv graphs_new_measurements output/figures/graphs_new_measurements
git mv plots output/figures/plots
```

- [ ] **Step 4: Commit**

```bash
git commit -m "chore: flyt billeder og grafer til output/figures/"
```

---

### Task 7: Flyt logfiler til output/logs/

**Files:**
- Move: `benchmark_run_stderr.log` → `output/logs/`
- Move: `benchmark_run_stdout.log` → `output/logs/`
- Move: `start_cmd_import.log` → `output/logs/`
- Move: `start_import_stderr.log` → `output/logs/`
- Move: `start_import_stdout.log` → `output/logs/`
- Move: `start_test_stderr.log` → `output/logs/`
- Move: `start_test_stdout.log` → `output/logs/`

- [ ] **Step 1: Opret output/logs/ og flyt filer**

```bash
mkdir -p output/logs
git mv benchmark_run_stderr.log output/logs/
git mv benchmark_run_stdout.log output/logs/
git mv start_cmd_import.log output/logs/
git mv start_import_stderr.log output/logs/
git mv start_import_stdout.log output/logs/
git mv start_test_stderr.log output/logs/
git mv start_test_stdout.log output/logs/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: flyt logfiler til output/logs/"
```

---

### Task 8: Fjern __pycache__ fra git tracking

**Files:**
- Remove: `__pycache__/` (4 .pyc-filer sporet af git)
- Remove: `tests/__pycache__/` (1 .pyc-fil sporet af git)

- [ ] **Step 1: Fjern fra git (sletter ikke de faktiske filer på disk)**

```bash
git rm -r --cached __pycache__/
git rm -r --cached tests/__pycache__/
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: fjern __pycache__ fra git tracking"
```

---

### Task 9: Opret .gitignore

**Files:**
- Create: `.gitignore`

- [ ] **Step 1: Skriv .gitignore**

Opret `.gitignore` i roden med præcis dette indhold:

```
output/
data/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: tilføj .gitignore"
```

---

### Task 10: Endelig verifikation

- [ ] **Step 1: Kør tests**

```bash
pytest tests/test_optimize_c2f2.py -v
```

Forventet: alle tests PASS.

- [ ] **Step 2: Bekræft mappestruktur**

```bash
ls src/ scripts/ tests/ data/ output/figures/ output/logs/
```

Forventet output — mapper med filer:
- `src/`: optimize_c2f2.py, laser_polarization_optimizer.py, monitor.py, pol_cons.py, poli2.ipynb
- `scripts/`: generate_100_graphs.py, generate_new_measurement_graphs.py, plot_*.py, report_plots.py
- `tests/`: test_optimize_c2f2.py, test_loop.py, run_continuous_test.py
- `data/`: benchmark_raw.csv, benchmark_summary.csv, benchmark_summary_c1_c3.csv, drift.csv, monitor_log.csv, opt_results.csv
- `output/figures/`: benchmark_*.png, report_fig*.png, IMG_6423.HEIC, graphs_100/, graphs_new_measurements/, plots/
- `output/logs/`: benchmark_run_stderr.log, benchmark_run_stdout.log, start_*.log

- [ ] **Step 3: Bekræft rod er ren**

```bash
ls *.py *.csv *.png *.log 2>/dev/null
```

Forventet: ingen output (alle rod-filer er flyttet).
