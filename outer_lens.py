"""G3: a learned outer response lens for two DC-equivalent hidden materials.

One *outer* port pair, two hidden capacitive paths. An externally imposed
material-history change leaves both the DC map and instantaneous response
identical while changing intermediate-time response. No new learning law.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def paths(history):
    """Same DC and instantaneous interface totals, different relaxation modes."""
    if history == 'A':
        return [(1., 1.), (1., 1.)]
    if history == 'B':
        return [(.2, .2), (1.8, 1.8)]
    raise ValueError(history)


def sample_response(u, history, *, capacity=2., dt=.05):
    network = paths(history)
    xs = np.zeros(2)
    g_left = np.array([v[0] for v in network])
    g_right = np.array([v[1] for v in network])
    rate = (g_left + g_right) / capacity
    a = np.exp(-rate * dt)
    w = g_left / (g_left + g_right)
    y = np.empty(len(u))
    for t, voltage in enumerate(u):
        y[t] = np.sum(g_left * (voltage - xs))
        xs = a * xs + (1-a) * w * voltage
    return y


def dc_conductance(history):
    return sum(gl*gr/(gl+gr) for gl,gr in paths(history))


def feedthrough(history):
    return sum(gl for gl,gr in paths(history))


def train_arx(u,y,p):
    """Fit ARX(p,p) from observed boundary inputs and outputs, 2p+1 parameters."""
    features = np.column_stack(
        [y[p-i:len(u)-i] for i in range(1,p+1)]
        + [u[p-j:len(u)-j] if j > 0 else u[p:] for j in range(p+1)])
    return np.linalg.lstsq(features,y[p:],rcond=None)[0]


def forecast_arx(u,theta,p):
    """Recursive held-out prediction; NEVER consume measured held-out outputs."""
    pred=np.zeros(len(u))
    for t in range(len(u)):
        pred[t] = sum(theta[i]*pred[t-i-1] for i in range(p) if t >= i+1)
        pred[t] += sum(theta[p+j]*u[t-j] for j in range(p+1) if t >= j)
    return pred


def train_fir(u,y,n):
    """Matched-coefficient-count non-recurrent FIR control."""
    x=np.column_stack([u[n-1-j:len(u)-j] if j else u[n-1:] for j in range(n)])
    return np.linalg.lstsq(x,y[n-1:],rcond=None)[0]


def forecast_fir(u,theta):
    n=len(theta)
    padded=np.pad(u,(n-1,0))
    return sum(theta[j]*padded[n-1-j:n-1-j+len(u)] for j in range(n))


def probes(seed,n,rate=10):
    rng=np.random.default_rng(seed)
    v=rng.uniform(-1,1,size=int(np.ceil(n/rate)))
    return np.repeat(v,rate)[:n] + .2*rng.standard_normal(n)


def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def run_gate(seed=20,train_size=2500,eval_size=800,train_noise=0.):
    histories=('A','B')
    trained={}
    for h in histories:
        u=probes(seed+100,train_size,rate=17)
        y=sample_response(u,h)
        train_rng=np.random.default_rng(seed+500+len(h)+ord(h[0]))
        y=y+train_noise*np.sqrt(np.mean(y*y))*train_rng.standard_normal(len(y))
        trained[h]={name:fit(u,y,order) for name,fit,order in (
            ('arx1',train_arx,1),('arx2',train_arx,2),
            ('fir3',train_fir,3),('fir5',train_fir,5))}
    heldout={
        'rapid':probes(seed+300,eval_size,rate=5),
        'slow':probes(seed+301,eval_size,rate=40),
        'wave':np.sin(.073*np.arange(eval_size))+.4*np.sin(.019*np.arange(eval_size)+1.5)
    }

    def predict(h,name,u):
        if name=='dc':
            return dc_conductance(h)*u
        if name.startswith('arx'):
            return forecast_arx(u,trained[h][name],int(name[-1]))
        return forecast_fir(u,trained[h][name])

    evaluation={}
    for drive,u in heldout.items():
        for h in histories:
            y=sample_response(u,h)
            evaluation[f'{h}:{drive}']={
                name+'_rmse':rmse(y,predict(h,name,u))
                for name in ('dc','arx1','arx2','fir3','fir5')
            }

    # Known-candidate identification. All predictions are learned using
    # TRAINING input-output pairs and generated for a HELDOUT input waveform.
    classif={}
    u=heldout['rapid']
    models=('dc','arx1','arx2','fir3','fir5')
    templates={name:np.stack([predict(h,name,u) for h in histories]) for name in models}
    trials=200
    truth=np.arange(trials)%2
    truth_y=np.stack([sample_response(u,h) for h in histories])[truth]
    sigma=np.sqrt(np.mean(truth_y**2,axis=1))
    for noise_fraction in (.001,.01,.05,.15):
        rng=np.random.default_rng(seed+501)
        observed=truth_y+noise_fraction*sigma[:,None]*rng.standard_normal(truth_y.shape)
        classif[str(noise_fraction)]={}
        for name in models:
            distances=np.mean((observed[:,None,:]-templates[name][None,:,:])**2,axis=-1)
            classif[str(noise_fraction)][name]=float(np.mean(distances.argmin(axis=1)==truth))

    return {
        'protocol':'Two boundary ports; two hidden capacitors in parallel paths; right grounded.',
        'histories':{h:{'paths':paths(h),'DC_conductance':dc_conductance(h),
                        'instantaneous_feedthrough':feedthrough(h)} for h in histories},
        'train_size':train_size,'eval_size':eval_size,'train_noise':train_noise,
        'parameters':{'dc':1,'fir3':3,'arx1':3,'fir5':5,'arx2':5},
        'coefficients':{h:{k:v.tolist() for k,v in trained[h].items()} for h in histories},
        'test':evaluation,
        'classification_accuracy':classif,
        'warnings':[
            'Histories are imposed conductance patterns, not autonomously learned.',
            'Same DC and instantaneous boundary maps by construction.',
            'Classical ARX / FIR learned from observed boundary training probes.',
            'ARX does not receive true prior outputs in the heldout evaluation.',
            'Decoder knows both candidates and K=1; the same long waveform is used in 200 noisy trials.',
            'Equal coefficient counts do NOT equalize recursive runtime state or compute.',
            'No new AI learning, capacity or efficiency claim.'
        ]
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/outer_lens.json'))
    args=p.parse_args()
    result=run_gate()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('DC and immediate response:', result['histories'])
    for k,row in result['test'].items():
        print(k,'ARX2',round(row['arx2_rmse'],6),
              'ARX1',round(row['arx1_rmse'],6),
              'FIR5',round(row['fir5_rmse'],6),
              'DC',round(row['dc_rmse'],6))
    print('History classification:',result['classification_accuracy'])
    print('Saved:',args.out)


if __name__=='__main__':
    main()
