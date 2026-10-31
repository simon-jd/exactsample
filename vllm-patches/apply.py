#!/usr/bin/env python3
"""Apply/reverse the bounded exactsample patch only to byte-verified v0.31.0 sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="vLLM checkout root or site-packages containing vllm/")
    parser.add_argument("--check", action="store_true", help="verify without changing files")
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args(argv)
    directory = Path(__file__).resolve().parent
    manifest = json.loads((directory / "manifest.json").read_text())
    patch = directory / manifest["patch"]
    if hashlib.sha256(patch.read_bytes()).hexdigest() != manifest["patch_sha256"]:
        parser.error("patch hash differs from manifest")
    before = "after_sha256" if args.reverse else "before_sha256"
    after = "before_sha256" if args.reverse else "after_sha256"
    for relative, record in manifest["files"].items():
        path = args.root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record[before]:
            parser.error(f"expected exact {manifest['version']} {'patched' if args.reverse else 'original'} source: {path}")
    command = ["git", "apply", "--check"]
    if args.reverse:
        command.append("--reverse")
    subprocess.run([*command, str(patch)], cwd=args.root.resolve(), check=True)
    if args.check:
        print("PASS: pinned source hashes and git apply --check")
        return
    command.remove("--check")
    subprocess.run([*command, str(patch)], cwd=args.root.resolve(), check=True)
    for relative, record in manifest["files"].items():
        if hashlib.sha256((args.root / relative).read_bytes()).hexdigest() != record[after]:
            raise RuntimeError(f"post-application hash mismatch: {relative}")
    print("Reversed" if args.reverse else "Applied", manifest["patch"])


if __name__ == "__main__":
    main()
