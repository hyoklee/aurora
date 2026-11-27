import h5py,numpy as np
f=h5py.File('/tmp/claude-36126/-home-hyoklee-aurora/7e1b65af-7054-4a75-8def-00dce7042087/scratchpad/qick097/agent097_beam-1.0-fp8/ss_ge.h5','r')
def load(g,key):
    return np.fromstring(g[key][()][0].decode().replace('[','').replace(']','').replace('\n',' '),sep=' ')
for q in ['Q1','Q2','Q3','Q4','Q5','Q6']:
    g=f[q]
    Ig=load(g,'I_g'); Qg=load(g,'Q_g'); Ie=load(g,'I_e'); Qe=load(g,'Q_e')
    ang=g['Angle'][()][0]
    # use stored angle, sign that gives larger dmu? test both signs, pick the one matching paper SNR ~4.4
    for sgn in [1,-1]:
        rg=Ig*np.cos(ang)+sgn*Qg*np.sin(ang)
        re=Ie*np.cos(ang)+sgn*Qe*np.sin(ang)
        mg,me=np.median(rg),np.median(re)
        if mg<me:
            gc=rg[rg<=mg]; ec=re[re>=me]; accfun=lambda rg,re,t: (np.sum(rg<=t)+np.sum(re>t))/(len(rg)+len(re))
        else:
            gc=rg[rg>=mg]; ec=re[re<=me]; accfun=lambda rg,re,t: (np.sum(rg>=t)+np.sum(re<t))/(len(rg)+len(re))
        sg=np.sqrt(np.mean((gc-mg)**2)); se=np.sqrt(np.mean((ec-me)**2))
        sbar=np.sqrt((sg*sg+se*se)/2); snr=abs(me-mg)/sbar
        vals=np.sort(np.concatenate([rg,re])); best=0; bestt=None
        for t in vals:
            acc=accfun(rg,re,t)
            if acc>best: best=acc; bestt=t
        print(f'{q} sgn={sgn}: storedF={g["Fidelity"][()][0]:.4f} SNR={snr:.3f} F_maxcontrast={best:.4f} (thr={bestt:.1f}) dmu={abs(me-mg):.1f} sbar={sbar:.2f}')
