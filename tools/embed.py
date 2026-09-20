"""Embed the compiled Metal library so wheels need no runtime compiler or paths."""
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()
lines = [", ".join(str(v) for v in data[i:i + 24]) for i in range(0, len(data), 24)]
Path(sys.argv[2]).write_text(
    "#pragma once\n#include <cstddef>\nnamespace metalgb {\n"
    "inline constexpr unsigned char kMetalLibrary[] = {\n"
    + ",\n".join(lines)
    + "\n};\ninline constexpr std::size_t kMetalLibrarySize = sizeof(kMetalLibrary);\n}\n"
)
