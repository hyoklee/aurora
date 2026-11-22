import h5py, re

with h5py.File("ss_ge.h5", "r") as f:
    ec = f["Q1/Exp Config"][0]
    if isinstance(ec, bytes):
        ec = ec.decode()

keys = re.findall(r"'([^']+)':", ec)
print("ALL KEYS:", keys)
print()
for kw in ("ss_ge", "fid", "Fid", "thresh", "angle", "Angle", "ro_", "integ"):
    for k in keys:
        if kw.lower() in k.lower():
            # print the key and a chunk of its value
            m = re.search(r"'%s':\s*(.{0,400})" % re.escape(k), ec, re.S)
            print("KEY %r -> %s" % (k, m.group(1)[:400] if m else "?"))
            print()
