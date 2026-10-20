"""Download and verify the data and checkpoints attached to a GitHub release."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(value):
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Non-relative asset path: {value}")
    target = ROOT.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(ROOT):
        raise ValueError(f"Asset path leaves repository: {value}")
    return target


def extract(archive):
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            target = local_path(info.filename)
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink entry in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with handle.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if target.exists() and sha256(target) != sha256(temporary):
                temporary.unlink()
                raise FileExistsError(f"Existing file differs from the release: {target}")
            os.replace(temporary, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="GitHub OWNER/REPOSITORY")
    parser.add_argument("--tag", default="v1.0.0")
    parser.add_argument("--groups", nargs="+", default=["all"],
                        choices=["all", "weights", "internal"])
    parser.add_argument("--seed", type=int, choices=range(1001, 1007),
                        help="Restrict weight downloads to one seed")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repo):
        parser.error("--repo must be OWNER/REPOSITORY")
    manifest = json.loads((ROOT / "release_assets_manifest.json").read_text())
    for item in manifest["assets"]:
        if "all" not in args.groups and item["group"] not in args.groups:
            continue
        if item["group"] == "weights" and args.seed is not None and item["seed"] != args.seed:
            continue
        target = local_path(item["local_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256(target) != item["sha256"]:
                raise FileExistsError(f"Existing file differs from the recorded asset: {target}")
            print(f"Verified existing asset: {item['name']}", flush=True)
        else:
            url = (f"https://github.com/{args.repo}/releases/download/"
                   f"{urllib.parse.quote(args.tag, safe='')}/{urllib.parse.quote(item['name'])}")
            temporary = target.with_name(target.name + ".part")
            print(f"Downloading {item['name']} ({item['bytes'] / 1024**2:.1f} MiB)", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Wound-release-downloader"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
            if temporary.stat().st_size != item["bytes"] or sha256(temporary) != item["sha256"]:
                raise ValueError(f"Asset verification failed: {temporary}")
            os.replace(temporary, target)
        if item["extract"]:
            extract(target)
    print("Requested release assets are present and verified.")


if __name__ == "__main__":
    main()
