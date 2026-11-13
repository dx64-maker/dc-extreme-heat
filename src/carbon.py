"""
Carbon / energy tracking with CodeCarbon (https://docs.codecarbon.io/).

Each pipeline stage (preprocessing, tuning, final training, batch inference)
is wrapped in `track("stage")`. CodeCarbon appends one row per stage to
outputs/emissions.csv, so failed or repeated runs accumulate in the same file
(the proposal counts unsuccessful runs too). If CodeCarbon is not installed
the pipeline still runs and records wall-clock time only.
"""
import contextlib
import time

from src import config

try:
    from codecarbon import OfflineEmissionsTracker
    HAVE_CODECARBON = True
except ImportError:  # pipeline must still work end-to-end without it
    HAVE_CODECARBON = False


@contextlib.contextmanager
def track(stage: str):
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tracker = None
    if HAVE_CODECARBON:
        tracker = OfflineEmissionsTracker(
            project_name=f"dc-heat:{stage}", country_iso_code="USA",
            output_dir=str(config.OUTPUT_DIR), log_level="error", save_to_file=True)
        tracker.start()
    try:
        yield
    finally:
        kg = tracker.stop() if tracker else None
        secs = time.time() - t0
        with open(config.OUTPUT_DIR / "runtime_log.csv", "a") as fh:
            fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{stage},{secs:.2f},"
                     f"{'' if kg is None else f'{kg:.8f}'}\n")
        print(f"[carbon] {stage}: {secs:.1f}s" + (f", {kg:.2e} kg CO2e" if kg is not None else
                                                   " (CodeCarbon not installed)"))
