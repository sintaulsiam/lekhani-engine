#!/usr/bin/env python3
"""
Mine high-confidence phonetic transliteration overrides from Dakshina lexicon datasets.
Invokes the high-performance native Rust builder in lekhani-core which applies
collision guards (protecting native Avro words like 'bal', 'fal', 'bay'),
Bayesian confidence calibration, and dual-candidate ranking.
"""

import subprocess
import sys
from pathlib import Path

def main():
    base_dir = Path(__file__).resolve().parent.parent
    cmd = ["cargo", "run", "--release", "-p", "lekhani-core", "--bin", "build_overrides"]
    print(f"🚀 Running: {' '.join(cmd)} in {base_dir}")
    res = subprocess.run(cmd, cwd=base_dir)
    sys.exit(res.returncode)

if __name__ == "__main__":
    main()
