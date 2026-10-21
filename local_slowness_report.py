"""Plot recorded local-slowness outcomes. Optional: pip install matplotlib."""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path
import numpy as np
from local_slowness_gate import load_receipt,ARMS


def render(receipt_path,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=load_receipt(receipt_path);summary=result['summary'];p=result['effective_config']
    out.parent.mkdir(parents=True,exist_ok=True)
    targets=[out,out.parent/'results.svg',out.parent/'results.png',out.parent/'material.svg',out.parent/'material.png']
    if any(path.exists() for path in targets):raise ValueError('refusing to overwrite report artifacts')
    colors={'local':'#2667b6','trace':'#e68a18','noise_off':'#18835e'}
    names={'local':'Local feedback','trace':'Equal-state trace','uniform':'Uniform / erased',
        'shuffled':'Shuffled material','noise_off':'No background noise','noise_only':'Noise without cue'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    delays=p['evaluation']['retention_delays']
    for arm in colors:
        axes[0].plot(delays,[summary['mean_accuracy'][f'retention:{d:g}'][arm] for d in delays],
            marker='o',lw=2.5,label=names[arm],color=colors[arm])
    axes[0].axhline(1/3,ls=':',color='gray',label='Three-class chance')
    axes[0].set(xlabel='Waiting time (simulation units)',ylabel='Old-history accuracy',ylim=(0,1.05),title='Retaining the first history')
    axes[0].legend(loc='lower left',fontsize=9)
    condition=f"rewrite:{p['evaluation']['primary_rewrite_delay']:g}"
    x=np.arange(3);arms=list(colors)
    new=[summary['mean_accuracy'][condition][a] for a in arms]
    old=[float(np.mean([d['conditions'][condition][a]['old_class_accuracy'] for d in result['devices']])) for a in arms]
    axes[1].bar(x,new,color=[colors[a] for a in arms],label='Reads replacement history')
    axes[1].bar(x,old,bottom=new,color='#c5c9cf',label='Still reads old history')
    axes[1].set(xticks=x,xticklabels=['Local','Trace','Noise off'],ylim=(0,1.05),ylabel='Fraction of decisions',title='After an equally long replacement write')
    axes[1].axhline(1/3,ls=':',color='gray');axes[1].legend(loc='lower right',fontsize=9)
    fig.suptitle(f"{len(result['devices'])} devices · fixed readout · fast activity cleared",fontsize=14)
    fig.savefig(targets[1]);fig.savefig(targets[2],dpi=160);plt.close(fig)

    device=result['devices'][0];size=p['model']['size']
    conditions=['retention:0',f"retention:{p['evaluation']['primary_delay']:g}",condition]
    fields=[np.asarray(device['example_materials'][c][a][0]).reshape(size,size) for a in ('local','trace') for c in conditions]
    fig,axes=plt.subplots(2,3,figsize=(10,6),layout='constrained')
    vmax=max(float(m.max()) for m in fields)
    for i,(ax,m) in enumerate(zip(axes.flat,fields)):
        image=ax.imshow(m,cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
        ax.set_xticks(range(size));ax.set_yticks(range(size))
        if i<3:ax.set_title(['After first history','After long waiting','After replacement'][i])
        if i%3==0:ax.set_ylabel(['Local feedback','Equal-state trace'][i//3])
    fig.colorbar(image,ax=axes,label='Stored material m; local rate = 1/(1+m)',shrink=.85)
    replacement=device['conditions'][condition]['local']['target_classes'][0]
    fig.suptitle(f"First device {device['seed']}, first example: history 0 → replacement {replacement}\nSame material total in every panel; no fast voltage remains",fontsize=12)
    fig.savefig(targets[3]);fig.savefig(targets[4],dpi=160);plt.close(fig)
    svgs=[path.read_text()[path.read_text().find('<svg'):] for path in (targets[1],targets[3])]
    rows=[]
    for arm in ARMS:
        cells=''.join(f"<td>{summary['mean_accuracy'][c][arm]:.2%}</td>" for c in summary['mean_accuracy'])
        rows.append(f"<tr><th>{names[arm]}</th>{cells}</tr>")
    gates=html.escape(json.dumps(summary['gates'],indent=2))
    headers=''.join('<th>'+html.escape(c)+'</th>' for c in summary['mean_accuracy'])
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainBoxSol — local slowness</title><style>body{font:17px/1.6 system-ui;max-width:1100px;margin:40px auto;padding:0 22px;color:#202d3b;background:#f7f9fc}h1{line-height:1.15}svg{max-width:100%;height:auto;background:white;border-radius:12px}section,pre{background:white;padding:22px;border-radius:12px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce2eb;text-align:right}th:first-child{text-align:left}pre{white-space:pre-wrap}small{color:#536579}</style>
<h1>Can a memory learn to change more slowly?</h1>'''
    content+=f"<p>Mode: <b>{result['mode']}</b>. {len(result['devices'])} device means; three classes; fixed calibration before waiting and replacement. Lower update speed is explicitly engineered.</p>"
    content+=svgs[0]+f"<section><h2>All controls</h2><div style='overflow:auto'><table><tr><th>Arm</th>{headers}</tr>{''.join(rows)}</table></div></section>"
    content+=f"<p>Baseline-adjusted retention gain: <b>{summary['retention_gain_local_minus_trace']:+.2%}</b>. Replacement accuracy cost: <b>{summary['rewrite_accuracy_cost']:.2%}</b>. Intervals resample device means, not the nine read samples.</p>"
    content+=svgs[1]+f"<h2>Locked gates</h2><pre>{gates}</pre>"
    content+="<section><h2>What this establishes</h2><p>The writer has no labels, reward, teacher or adjoint. The calibrated readout is supervised. Each primary arm stores 25 material scalars; global resource normalization and read-phase freezing are explicit controls. Persistent material survives a fast-state reset. This does not establish physical mass, time dilation, noise necessity, autonomous learning, greater capacity or digital efficiency.</p><p>The uniform arm also erases the spatial trace. Shuffling preserves its values while breaking their locations. Noise-off changes input power, so its comparison is diagnostic only. The fixed conductance operator is unchanged and its DC solution cannot be repaired by mobility alone.</p></section>"
    content+=f"<small>Source/protocol lock: {html.escape(str(result['lock_commit']))}. Receipts: {html.escape(receipt_path.name)} and adjacent device files; separate full-audit.json. Writing, waiting, replacement and calibration are regenerated in the replay audit.</small></html>"
    with out.open('x') as stream:stream.write(content)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--receipt',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();render(args.receipt,args.out)


if __name__=='__main__':main()
