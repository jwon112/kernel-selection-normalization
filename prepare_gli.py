#!/usr/bin/env python3
"""GLI-only portable entry point reusing the unchanged study preprocessing functions."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    root = here if (here / "src").is_dir() else here.parents[2]
    sys.path.insert(0, str(root / "src"))
    import prepare_brats_multidomain_v2_anatomical as anatomical
    import prepare_brats2023_gli_benchmark as benchmark

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--cache-out", type=Path, required=True)
    parser.add_argument("--benchmark-out", type=Path, required=True)
    args = parser.parse_args()
    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    if args.cache_out.exists() or args.benchmark_out.exists():
        raise FileExistsError("Both output paths must be new; no cache is overwritten")
    anatomical.common.ARCHIVES = {"brats2023_gli": [args.archive.resolve()]}
    # The anatomical implementation fixes its own (192,224,192) canvas; the
    # older common module's center-crop function/canvas is never called here.
    for name in ("images", "labels_pretreatment_3class"):
        (args.cache_out / name).mkdir(parents=True, exist_ok=False)
    records, transforms = [], {}
    anatomical.process_2023(args.cache_out, records, transforms)
    records.sort(key=lambda r: (r["domain"], r["case_id"]))
    if len(records) != 1251:
        raise ValueError("The archive does not contain the recorded 1251-study cohort")
    (args.cache_out / "cases.json").write_text(json.dumps(records, indent=2) + "\n")
    (args.cache_out / "spatial_transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    subprocess.run([sys.executable, str(Path(benchmark.__file__)), "--source", str(args.cache_out.resolve()),
                    "--out", str(args.benchmark_out.resolve()), "--seed", "2026"], check=True)
    expected = json.loads((root / "data_contract/splits.json").read_text())
    actual = json.loads((args.benchmark_out / "splits.json").read_text())
    if actual != expected:
        raise ValueError("Regenerated split membership differs from the study snapshot")
    print("PASS: recorded split membership; anatomical crop margin 4, pad only, nonzero z-score")


if __name__ == "__main__":
    main()
