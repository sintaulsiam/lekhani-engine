#!/usr/bin/env python3
"""
Compile supervised phonetic overrides JSON into a compact binary table.
Format:
  [4 bytes: b"POVR"] [4 bytes: version (1)]
  [4 bytes: entry_count]
  Repeated entries:
    [2 bytes: latin_len] [latin_bytes]
    [2 bytes: cand_count]
    Repeated candidates:
      [2 bytes: bengali_len] [bengali_bytes]
      [4 bytes: float32 confidence]
"""

import json
import struct
import sys
from pathlib import Path

def compile_overrides(json_path: Path, bin_path: Path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    out = bytearray()
    out.extend(b"POVR")
    out.extend(struct.pack("<I", 1)) # Version 1
    out.extend(struct.pack("<I", len(data)))

    for latin, candidates in data.items():
        latin_bytes = latin.encode('utf-8')
        out.extend(struct.pack("<H", len(latin_bytes)))
        out.extend(latin_bytes)

        out.extend(struct.pack("<H", len(candidates)))
        for item in candidates:
            bengali, conf = item[0], float(item[1])
            bengali_bytes = bengali.encode('utf-8')
            out.extend(struct.pack("<H", len(bengali_bytes)))
            out.extend(bengali_bytes)
            out.extend(struct.pack("<f", conf))

    with open(bin_path, 'wb') as f:
        f.write(out)

    print(f"✅ Compiled {len(data)} overrides from {json_path} -> {bin_path} ({len(out)} bytes)")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    src_json = base_dir / "data" / "dictionaries" / "phonetic_overrides.json"
    dst_bin = base_dir / "data" / "dictionaries" / "phonetic_overrides.bin"

    if len(sys.argv) > 1:
        src_json = Path(sys.argv[1])
    if len(sys.argv) > 2:
        dst_bin = Path(sys.argv[2])

    compile_overrides(src_json, dst_bin)
