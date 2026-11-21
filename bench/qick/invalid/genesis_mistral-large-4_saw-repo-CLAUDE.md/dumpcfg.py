import sys
import h5py

path = sys.argv[1]
f = h5py.File(path, "r")
g = f["Q1"]
ec = g["Exp Config"][0].decode()
i = ec.find("ss_ge")
print("=== EXP CONFIG around ss_ge ===")
print(ec[max(0, i - 80): i + 2000])
print()
print("=== SYST CONFIG (first 2500 chars) ===")
print(g["Syst Config"][0].decode()[:2500])
