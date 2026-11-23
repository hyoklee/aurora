import sys
sys.path.insert(0, '/home/hyoklee/clio-core.dataverse/build/bin')
import cte_golf as cg

cg.connect()
print('Connected OK')
m = cg.manifest()
print(f'Manifest has {len(m)} rows')
for r in m[:10]:
    print(r)
