"""Build reproducible release archives and record all published file hashes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", "__pycache__", ".venv", ".downloads", "outputs"}


def files_below(folder):
    return [p for p in sorted(folder.rglob("*")) if p.is_file()
            and not any(part in IGNORED_PARTS for part in p.relative_to(ROOT).parts)
            and p.suffix not in (".pyc", ".part") and p.name != ".DS_Store"]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(name, folders):
    target = ROOT / "release_assets" / name
    members = [p for folder in folders for p in files_below(ROOT / folder)]
    if not members:
        raise ValueError(f"No source files for {name}")
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in members:
            if path.is_symlink():
                raise ValueError(f"Do not archive external links: {path}")
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), (2026, 10, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as source, handle.open(info, "w", force_zip64=True) as dest:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    dest.write(block)
    if target.stat().st_size >= 2 * 1024**3:
        raise ValueError(f"Release asset exceeds 2 GiB: {name}")
    return target


def published_files():
    with tempfile.TemporaryDirectory(prefix="wound_release_inventory_") as temporary:
        git_dir = Path(temporary) / "git"
        subprocess.run(["git", "init", "--bare", "--quiet", str(git_dir)], check=True)
        output = subprocess.check_output([
            "git", "--git-dir", str(git_dir), "--work-tree", str(ROOT),
            "ls-files", "--others", "--exclude-standard", "-z",
        ])
    paths = {value.decode() for value in output.split(b"\0") if value}
    manifest = json.loads((ROOT / "release_assets_manifest.json").read_text())
    for item in manifest["assets"]:
        paths.add(item["local_path"])
        if item["extract"]:
            with zipfile.ZipFile(ROOT / item["local_path"]) as handle:
                paths.update(info.filename for info in handle.infolist() if not info.is_dir())
    paths.discard("MANIFEST.sha256")
    return [ROOT / relative for relative in sorted(paths)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    (ROOT / "release_assets").mkdir(exist_ok=True)
    if not args.inventory_only:
        assets = []
        for seed in range(1001, 1007):
            path = ROOT / "weights" / f"seed_{seed}.pth"
            if not path.is_file():
                raise FileNotFoundError(path)
            assets.append(dict(name=path.name, local_path=path.relative_to(ROOT).as_posix(),
                               group="weights", seed=seed, extract=False, bytes=path.stat().st_size,
                               sha256=sha256(path)))
        for name, group, folders in (
            ("internal_dataset.zip", "internal", ["datasets/internal"]),
        ):
            print(f"Preparing {name}", flush=True)
            path = archive(name, folders)
            assets.append(dict(name=name, local_path=path.relative_to(ROOT).as_posix(), group=group,
                               extract=True, bytes=path.stat().st_size, sha256=sha256(path)))
        (ROOT / "release_assets_manifest.json").write_text(
            json.dumps({"tag": "v1.0.0", "assets": assets}, indent=2) + "\n")
    lines = []
    for index, path in enumerate(published_files(), 1):
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n")
        if index % 250 == 0:
            print(f"Inventoried {index} files", flush=True)
    (ROOT / "MANIFEST.sha256").write_text("".join(lines))
    print(json.dumps({"files": len(lines), "manifest": "MANIFEST.sha256"}))


if __name__ == "__main__":
    main()
