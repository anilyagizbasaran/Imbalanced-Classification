"""
Generates stepwise BOD visualizations for all 12 KEEL datasets.
Output: result_images/plots_<dataset>/
"""

import os
import argparse
from stepwise_visualizer import run_stepwise_viz

DATASETS = [
    "glass1", "yeast1", "haberman", "ecoli1", "segment0", "glass6",
    "yeast2vs4", "glass0146vs2", "yeast1vs7", "glass4", "yeast5", "yeast6",
]

OUTPUT_DIR = "result_images"

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    failed = []
    for ds in DATASETS:
        print(f"\n{'#'*60}\n# Dataset: {ds}\n{'#'*60}")
        args = argparse.Namespace(
            dataset=ds,
            datadir="./datasets",
            k=5,
            min_samples=5,
            outdir=OUTPUT_DIR,
        )
        try:
            run_stepwise_viz(args)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            failed.append((ds, str(e)))

    print(f"\n{'='*60}")
    print(f"Done. {len(DATASETS) - len(failed)}/{len(DATASETS)} datasets OK.")
    if failed:
        for ds, err in failed:
            print(f"  FAILED: {ds} -- {err}")
