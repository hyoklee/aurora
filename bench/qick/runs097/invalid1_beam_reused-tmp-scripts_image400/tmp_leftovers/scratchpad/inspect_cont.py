import numpy as np
x=np.linspace(0,1,10); y=np.linspace(0,1,10)
X,Y=np.meshgrid(x,y)
Z=X+Y
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
levels=[0.5,1.0,1000]
cf=plt.contourf(X,Y,Z,levels=levels,colors=['indigo','mediumseagreen','b'])
print(type(cf))
print('levels', cf.levels)
print('allsegs len', len(cf.allsegs) if hasattr(cf,'allsegs') else None)
print('collections attr?', hasattr(cf,'collections'))
print('legend_elements', cf.legend_elements())
for c in cf.collections if hasattr(cf,'collections') else []:
    print('coll', c.get_facecolor())
plt.close()
