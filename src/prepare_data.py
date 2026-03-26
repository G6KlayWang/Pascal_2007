from __future__ import annotations

import argparse

from src.config import apply_overrides, load_config
from src.data import build_data_overview, build_internal_split, maybe_download_voc
from src.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare Pascal VOC 2007.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    output_dir = ensure_dir(config["experiment"]["output_root"]) / "data_overview"
    voc_root = maybe_download_voc(config["dataset"]["root"], config["dataset"].get("download", False))
    split_manifest = build_internal_split(
        voc_root=voc_root,
        output_dir=output_dir,
        val_ratio=config["dataset"]["internal_val_ratio"],
        seed=config["dataset"]["split_seed"],
    )
    build_data_overview(voc_root, split_manifest, output_dir)
    print(f"Prepared Pascal VOC 2007 under {voc_root}")
    print(f"Saved overview artifacts to {output_dir}")


if __name__ == "__main__":
    main()
