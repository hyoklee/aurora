import h5py, numpy as np, ast, pprint
f=h5py.File('ss_ge.h5','r')
g=f['Q1']
ec=g['Exp Config'][0].decode('utf-8','replace')
print('EXP CONFIG repr len', len(ec))
# find ss_ge section
i=ec.find('ss_ge')
print(ec[i-200:i+1500] if i>=0 else 'no ss_ge')
