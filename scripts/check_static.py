#!/usr/bin/env python3
"""Check source artifacts without importing any experiment or ML library."""
import ast
import hashlib
import io
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    sources = sorted(ROOT.rglob("*.py"))
    sources = [p for p in sources if not any(x in p.parts for x in (".venv", "__pycache__"))]
    for path in sources:
        tree = ast.parse(path.read_text(), filename=str(path))
        compile(tree, str(path), "exec")  # Compile only; no exec/import of experiment code.
    for path in ROOT.joinpath("configs").glob("*.json"):
        assert isinstance(json.loads(path.read_text()), dict), path
    primary = ast.parse((ROOT / "dtb/training/cifar10_resnet18_stage3.py").read_text())
    config_class = next(n for n in primary.body if isinstance(n, ast.ClassDef) and n.name == "Stage3Cfg")
    config_fields = {n.target.id for n in config_class.body if isinstance(n, ast.AnnAssign)}
    for path in ROOT.joinpath("configs").glob("*.json"):
        if path.name == "imagenet12_classes.json":
            continue
        unknown = set(json.loads(path.read_text())) - config_fields
        assert not unknown, f"Unknown primary Stage 3 fields in {path}: {unknown}"
    manifest = json.loads((ROOT / "docs/source_manifest.json").read_text())
    for item in manifest:
        path = ROOT / item["destination"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == item["release_sha256"], f"Manifest hash mismatch: {path}"
    for path in ROOT.joinpath("scripts").glob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
    try:
        from pyflakes.api import check
        from pyflakes.reporter import Reporter
    except ImportError:
        print("Optional undefined-name scan skipped: install requirements-dev.txt to enable it.")
    else:
        units = [(str(p), p.read_text()) for p in sources if not any(x in p.parts for x in ("defenses", "chain_sources"))]
        for folder in ("defenses", "chain_sources"):
            for path in ROOT.joinpath("dtb", folder).glob("*.py"):
                if path.stem == "__init__" or path.stem.endswith("_core"):
                    continue
                core = path.with_name(path.stem + "_core.py")
                units.append((str(path), 'data_path=""\nckpt_path=""\n' + core.read_text() + "\n" + path.read_text()))
        problems = []
        for filename, source in units:
            stream = io.StringIO()
            check(source, filename, Reporter(stream, stream))
            problems.extend(line for line in stream.getvalue().splitlines() if "undefined" in line or "syntax" in line)
        if problems:
            raise RuntimeError("\n".join(problems))
        print(f"Undefined-name scan passed for {len(units)} source units.")
    print(f"Static checks passed: {len(sources)} Python files; {len(manifest)} extracted-source hashes; JSON; bash -n.")
    print("No training, evaluation, data download, or ML imports were performed.")


if __name__ == "__main__":
    main()
