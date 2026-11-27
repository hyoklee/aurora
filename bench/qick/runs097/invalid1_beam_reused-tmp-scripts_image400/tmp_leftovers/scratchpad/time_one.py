import time, re
import cte_golf as cg
m=cg.manifest()
t0=time.time()
tags=cg.tags(r'golf/P01\.zip/.*')
data_tags=[t for t in tags if t.endswith('/data')]
print('P01 data tags', len(data_tags))
n=0; bytes=0
for t in data_tags:
    a=cg.read_array(t); n+=1; bytes+=a.nbytes
print('read', n, 'bytes', bytes, 'time', round(time.time()-t0,2))
