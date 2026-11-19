#!/usr/bin/env python3
"""Download or verify final model weights and area image and mask archives.

The adjacent final_release_assets.json supplies release URLs, destination paths,
sizes and SHA256 checksums. Examples:
  python download_final_assets.py
  python download_final_assets.py --group rat --seed 1005
  python download_final_assets.py --group rat --group human --verify-only

--seed filters weight files only; area archives remain selected unless
--group excludes it. Archives are downloaded without extraction. Matching files
are skipped; existing files with different contents are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tempfile
from urllib.error import URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


SCRIPT_DIR = Path(__file__).resolve().parent
CHUNK_SIZE = 1024 * 1024
GROUPS = ("rat", "human", "area")


class AssetError(ValueError):
    """An asset or its manifest cannot be used safely or verified."""


def destination_path(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while rejecting traversal and escaping symlinks."""
    if not isinstance(relative_path, str) or not relative_path:
        raise AssetError("relative_path must be a nonempty string")
    posix = PurePosixPath(relative_path)
    windows = PureWindowsPath(relative_path)
    if (posix.is_absolute() or windows.drive or "\\" in relative_path
            or "\x00" in relative_path
            or any(part in ("", ".", "..") for part in relative_path.split("/"))):
        raise AssetError(f"Unsafe destination path: {relative_path!r}")
    root = root.resolve()
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root) or target == root:
        raise AssetError(f"Destination escapes its root: {relative_path!r}")
    return target


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise AssetError("Manifest must be a JSON object")
    repository = manifest.get("repository")
    tag = manifest.get("release_tag")
    if not isinstance(repository, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise AssetError("Manifest repository must have the form owner/repository")
    if not isinstance(tag, str) or not tag:
        raise AssetError("Manifest release_tag must be a nonempty string")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise AssetError("Manifest assets must be a nonempty list")
    names, destinations = set(), set()
    for asset in assets:
        if not isinstance(asset, dict):
            raise AssetError("Every asset must be a JSON object")
        name = asset.get("name")
        if (not isinstance(name, str) or not name or name in (".", "..")
                or "/" in name or "\\" in name or "\x00" in name):
            raise AssetError("Asset name must be a plain filename")
        relative = asset.get("relative_path")
        destination_path(path.parent, relative)
        if name in names or relative in destinations:
            raise AssetError(f"Duplicate asset name or destination: {name}")
        names.add(name)
        destinations.add(relative)
        size = asset.get("size_bytes")
        if type(size) is not int or size < 0:
            raise AssetError(f"Invalid size_bytes for {name}")
        digest = asset.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise AssetError(f"Invalid SHA256 for {name}")
        group = asset.get("group")
        if group not in GROUPS:
            raise AssetError(f"Invalid group for {name}")
        if group != "area" and (type(asset.get("seed")) is not int
                                 or asset["seed"] not in range(1001, 1007)):
            raise AssetError(f"Invalid or missing training seed for {name}")
        expected_url = (f"https://github.com/{repository}/releases/download/"
                        f"{quote(tag, safe='')}/{quote(name, safe='')}")
        if asset.get("url") != expected_url:
            raise AssetError(f"URL is not the declared GitHub HTTPS release asset: {name}")
    return manifest


def select_assets(manifest: dict, groups=None, seed=None) -> list:
    selected_groups = set(groups or GROUPS)
    return [asset for asset in manifest["assets"]
            if asset["group"] in selected_groups
            and (seed is None or asset["group"] == "area" or asset["seed"] == seed)]


def verify_file(path: Path, asset: dict) -> None:
    if not path.is_file():
        raise AssetError(f"Missing file or not a regular file: {path}")
    size = path.stat().st_size
    if size != asset["size_bytes"]:
        raise AssetError(f"Size mismatch for {path}: expected {asset['size_bytes']}, got {size}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    if digest.hexdigest() != asset["sha256"]:
        raise AssetError(f"SHA256 mismatch for {path}; existing file was not overwritten")


class HTTPSRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        # GitHub release downloads redirect to HTTPS asset storage.
        parsed = urlsplit(new_url)
        if parsed.scheme.lower() != "https" or parsed.username or parsed.password:
            raise AssetError("Refusing a non-HTTPS or credential-bearing redirect")
        return super().redirect_request(request, fp, code, message, headers, new_url)


def open_download(url: str):
    if urlsplit(url).scheme.lower() != "https":
        raise AssetError("Downloads require HTTPS")
    request = Request(url, headers={"User-Agent": "woundseg-final-assets/1.0"})
    return build_opener(HTTPSRedirectHandler()).open(request, timeout=60)


def process_asset(asset: dict, root: Path, verify_only=False) -> str:
    target = destination_path(root, asset["relative_path"])
    if os.path.lexists(target):
        verify_file(target, asset)
        return "verified" if verify_only else "already present"
    if verify_only:
        raise AssetError(f"Missing file: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if destination_path(root, asset["relative_path"]) != target:
        raise AssetError("Destination changed while preparing the download")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".part",
                                         delete=False) as output:
            temporary = Path(output.name)
            digest, size = hashlib.sha256(), 0
            with open_download(asset["url"]) as response:
                for chunk in iter(lambda: response.read(CHUNK_SIZE), b""):
                    size += len(chunk)
                    if size > asset["size_bytes"]:
                        raise AssetError(f"Download exceeds expected size: {asset['name']}")
                    digest.update(chunk)
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if size != asset["size_bytes"] or digest.hexdigest() != asset["sha256"]:
            raise AssetError(f"Downloaded size or SHA256 mismatch: {asset['name']}")
        if destination_path(root, asset["relative_path"]) != target:
            raise AssetError("Destination changed during the download")
        # Publishing a hard link is atomic and cannot overwrite an existing path.
        # The temporary file is in the same directory, hence the same filesystem.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if destination_path(root, asset["relative_path"]) != target:
                raise AssetError("Destination changed during publication")
            verify_file(target, asset)
            return "already present"
        temporary.unlink()
        temporary = None
        return "downloaded"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=GROUPS, action="append",
                        help="Select a group; repeat to select several (default: all)")
    parser.add_argument("--seed", type=int, choices=range(1001, 1007),
                        help="Filter weight files by seed; does not filter area archives")
    parser.add_argument("--destination", type=Path, default=SCRIPT_DIR,
                        help="Destination root (default: directory containing this script)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Check selected local files without downloading or writing")
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(SCRIPT_DIR / "final_release_assets.json")
        assets = select_assets(manifest, args.group, args.seed)
        if not assets:
            raise AssetError("No assets match the requested selection")
        root = args.destination.resolve()
        # Validate all selected paths before writing any asset.
        for asset in assets:
            destination_path(root, asset["relative_path"])
        for asset in assets:
            print(f"Checking {asset['name']} ...", flush=True)
            status = process_asset(asset, root, args.verify_only)
            print(f"{status.capitalize()}: {asset['relative_path']}", flush=True)
        return 0
    except (AssetError, OSError, URLError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
