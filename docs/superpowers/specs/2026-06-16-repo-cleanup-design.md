# Repo Cleanup Design — 2026-06-16

## Mål

Omorganiser repository fra en flad struktur til en tydeligt opdelt mappestruktur med klare ansvarsområder. Den vigtigste kildekod er `optimize_c2f2.py`.

## Ny mappestruktur

```
src/
  optimize_c2f2.py              # primær optimeringskode
  laser_polarization_optimizer.py
  monitor.py
  pol_cons.py
  poli2.ipynb

scripts/
  generate_100_graphs.py
  generate_new_measurement_graphs.py
  plot_benchmark.py
  plot_calibration_time.py
  plot_drift_analysis.py
  plot_monitor.py
  plot_threshold.py
  report_plots.py

tests/
  test_optimize_c2f2.py         (allerede her)
  test_loop.py                  (flyttes fra rod)
  run_continuous_test.py        (flyttes fra rod)

data/
  benchmark_raw.csv
  benchmark_summary.csv
  benchmark_summary_c1_c3.csv
  drift.csv
  monitor_log.csv
  opt_results.csv

output/
  figures/
    graphs_100/                 (flyttes fra rod)
    graphs_new_measurements/    (flyttes fra rod)
    plots/                      (flyttes fra rod)
    benchmark_*.png             (flyttes fra rod)
    report_fig*.png             (flyttes fra rod)
    IMG_6423.HEIC               (flyttes fra rod)
  logs/
    benchmark_run_stderr.log
    benchmark_run_stdout.log
    start_cmd_import.log
    start_import_stderr.log
    start_import_stdout.log
    start_test_stderr.log
    start_test_stdout.log

conftest.py                     (i roden, opdateret til sys.path)
.gitignore                      (ny)
```

## Importhåndtering

`monitor.py` og `test_loop.py` importerer direkte `from optimize_c2f2 import ...`.  
`conftest.py` opdateres til at tilføje `src/` til `sys.path`, så alle moduler i `src/` kan importeres uden præfiks fra tests og scripts.

```python
# conftest.py (opdateret)
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
```

## Slettes

- `__pycache__/` og `tests/__pycache__/` (genererede filer)

## .gitignore

Ignorerer genereret output og Python-artefakter:

```
output/
data/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
```

## Ikke berørt

- `.claude/` (Claude Code indstillinger)
- `docs/` (dokumentation)
- `tests/test_optimize_c2f2.py` (allerede korrekt placeret)
