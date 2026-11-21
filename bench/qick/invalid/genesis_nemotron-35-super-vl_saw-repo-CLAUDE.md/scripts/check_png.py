import struct

import sys

with open(sys.argv[1] if len(sys.argv) > 1 else "readout.png", "rb") as f:
    sig = f.read(8)
    print("PNG signature ok:", sig == b"\x89PNG\r\n\x1a\n")
    f.read(4)  # chunk length
    f.read(4)  # "IHDR"
    w, h = struct.unpack(">II", f.read(8))
    print("dimensions:", w, "x", h)
