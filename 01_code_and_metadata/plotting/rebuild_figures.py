"""Rebuild replacement figures from the files bundled with this repository."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-local-photos", action="store_true",
        help="Also rebuild the photographic area review and placement panels from the complete local photo assets.",
    )
    args = parser.parse_args()
    area = ROOT / "analysis" / "wound_area_B_100"
    placement = ROOT / "analysis" / "placement" / "overlay_display"
    if args.include_local_photos:
        with (area / "data" / "measurements.csv").open(newline="") as handle:
            required = [area / row["overlay_path"] for row in csv.DictReader(handle)]
        records = json.loads((placement / "sources" / "overlay_records.json").read_text())
        required.extend(placement / row["source_image"] for row in records)
        missing = [path for path in required if not path.is_file()]
        if missing:
            parser.error(
                f"{len(missing)} local photographic inputs are missing; first: "
                f"{missing[0].relative_to(ROOT)}. Run without --include-local-photos "
                "to rebuild the numerical figures and preserve the supplied photographic SVG."
            )
    elif not (ROOT / "figures" / "FigureS22_placement_overlays.svg").is_file():
        parser.error("Restore figures/FigureS22_placement_overlays.svg from the repository before rebuilding.")

    for script in [
        HERE / "internal" / "plot_figures.py",
        HERE / "architecture" / "render_architecture.py",
        HERE / "workflow" / "render_workflow.py",
        HERE / "render_tables.py",
    ]:
        subprocess.run([sys.executable, str(script)], check=True)

    area_command = [sys.executable, str(area / "code" / "reproduce_area.py")]
    if not args.include_local_photos:
        area_command.append("--plots-only")
    subprocess.run(area_command, check=True)
    for stem in ["Figure2K_area_selected100", "Figure2L_area_selected100",
                 "Figure2KL_area_selected100"]:
        for ext in ["svg", "pdf", "png"]:
            shutil.copy2(area / "figures" / f"{stem}.{ext}", ROOT / "figures" / f"{stem}.{ext}")

    if args.include_local_photos:
        subprocess.run([sys.executable, str(placement / "build_overlay_display.py"),
                        "--root", str(placement)], check=True)
        for ext in ["svg", "pdf", "png"]:
            shutil.copy2(placement / "exports" / f"FigR_photographic_overlays.{ext}",
                         ROOT / "figures" / f"FigureS22_placement_overlays.{ext}")
    else:
        print("Preserved the supplied FigureS22_placement_overlays.svg with its embedded photographs.")


if __name__ == "__main__":
    main()
