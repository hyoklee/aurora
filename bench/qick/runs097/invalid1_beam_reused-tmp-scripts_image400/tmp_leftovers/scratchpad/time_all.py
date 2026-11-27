import time, re
import cte_golf as cg
m=cg.manifest()
t0=time.time()
alltags=[]
for r in m:
    # tags are not in manifest for streams; use tags() per zip? Instead iterate manifest paths and build tag names?
    pass
# Use cg.tags for each zip
zips=sorted(set(r['path'].split('/')[0] for r in m))
data_tags=[]
for z in zips:
    ts=cg.tags('golf/'+re.escape(z).replace('\\.','.')+'/.*')  # placeholder
# easier: regex escape dots
import re as _re
data_tags=[]
for z in zips:
    pat='golf/'+_re.escape(z).rstrip('.zip') if False else 'golf/'+z.replace('.','\\.')+'/.*'
    ts=cg.tags(pat)
    data_tags += [t for t in ts if t.endswith('/data')]
print('total data tags', len(data_tags))
t1=time.time()
n=0; bytes=0
for t in data_tags:
    a=cg.read_array(t); n+=1; bytes+=a.nbytes
t2=time.time()
print('tags listing time', round(t1-t0,2), 'read time', round(t2-t1,2), 'total', round(t2-t0,2))
print('bytes', bytes)
