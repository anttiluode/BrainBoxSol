"""Self-contained replay of measured medium states; does not train in-browser."""
from dataclasses import asdict
import html
import json
from pathlib import Path
import numpy as np
from adaptive_medium import make_grid, rollout
from adaptive_medium_task import TaskConfig, make_drive, read_motor, episode_from_energies
from adaptive_medium_gate import make_evaluation_sources


def replay_episodes(receipt):
    cfg=TaskConfig(**receipt['config']); grid=make_grid(cfg.size)
    row=receipt['seed_results'][0]; sources,_=make_evaluation_sources(row['seed'],cfg,1)
    records=[]; indices=np.unique(np.r_[np.arange(0,cfg.steps+1,4),cfg.steps])
    for mapping_record in row['mappings']:
        for arm,key in (('Before','initial_material'),('Learned','learned_material'),
                        ('Frozen','initial_material'),('Erased','initial_material')):
            g=np.asarray(mapping_record[key])
            for cue,source in enumerate(sources):
                states=rollout(grid,g,make_drive(source,cfg),dt=cfg.dt,leak=cfg.leak)
                ep=read_motor(states,cfg)
                records.append({'arm':arm,'cue':cue,'seed':row['seed'],'mapping':mapping_record['mapping'],
                    'target_motor':mapping_record['mapping'][cue],'material':g.tolist(),
                    'source':source.tolist(),'states':states[indices].tolist(),'times':(indices*cfg.dt).tolist(),
                    'energies':ep.energies.tolist(),'score':ep.score,'action':ep.action,'valid':ep.valid})
    return records


def write_replay(path,*,grid,receipt,episodes):
    if not episodes: raise ValueError('replay requires measured episodes')
    cfg=TaskConfig(**receipt['config']); combinations=set(); mappings=set()
    for ep in episodes:
        try:
            states=np.asarray(ep['states'],dtype=float); times=np.asarray(ep['times'],dtype=float)
            material=np.asarray(ep['material'],dtype=float)
            if (states.ndim!=2 or states.shape[1]!=grid.nodes or len(states)<2 or len(times)!=len(states)
                    or not np.isfinite(states).all() or not np.isfinite(times).all()
                    or np.any(np.diff(times)<=0) or times[0]!=0
                    or not np.isclose(times[-1],cfg.steps*cfg.dt)
                    or material.shape!=(len(grid.edges),) or not np.isfinite(material).all()):
                raise ValueError('inconsistent replay arrays')
            make_drive(ep['source'],cfg)
            measured=episode_from_energies(ep['energies'],cfg)
            if (ep['cue'] not in (0,1) or ep['arm'] not in ('Before','Learned','Frozen','Erased')
                    or ep['mapping'] not in ([0,1],[1,0]) or not isinstance(ep['seed'],int)
                    or ep['target_motor']!=ep['mapping'][ep['cue']]
                    or ep['action']!=measured.action or ep['valid']!=measured.valid
                    or not np.isfinite(ep['score']) or not np.isclose(ep['score'],measured.score)
                    or np.any(states<0) or np.any(states[0]!=0)
                    or material.min()<cfg.lower-1e-12 or material.max()>cfg.upper+1e-12
                    or abs(material.sum()-len(material)*cfg.mean)>1e-10*len(material)*cfg.mean):
                raise ValueError('inconsistent replay metadata or material')
            key=(tuple(ep['mapping']),ep['cue'],ep['arm'])
            if key in combinations: raise ValueError('duplicate replay condition')
            combinations.add(key);mappings.add(tuple(ep['mapping']))
        except (KeyError,TypeError) as exc:
            raise ValueError('missing replay arrays') from exc
    expected={(m,c,a) for m in mappings for c in (0,1) for a in ('Before','Learned','Frozen','Erased')}
    if combinations!=expected: raise ValueError('missing selectable replay condition')
    payload={'episodes':episodes,'size':grid.size,'edges':grid.edges.tolist(),
        'config':asdict(cfg),'commit':receipt['code_commit'],'mode':receipt['mode']}
    encoded=json.dumps(payload,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    rows=''.join(f'<tr><td>{html.escape(k)}</td><td>{v["mean"]:.3f}</td></tr>'
                 for k,v in receipt['summary']['arms'].items())
    page=TEMPLATE.replace('__DATA__',encoded).replace('__ROWS__',rows)
    options=''.join(f'<option value="{m[0]},{m[1]}">A → {"upper" if m[0]==0 else "lower"} · B → {"upper" if m[1]==0 else "lower"}</option>'
                    for m in sorted(mappings,reverse=True))
    page=page.replace('__MAP_OPTIONS__',options)
    page=page.replace('__MODE__',html.escape(receipt['mode'])).replace('__COMMIT__',html.escape(receipt['code_commit']))
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(page,encoding='utf-8')


TEMPLATE='''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BrainBoxSol · Experience changes the medium</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#10141e;color:#e7edf4}*{box-sizing:border-box}
body{margin:0 auto;max-width:1200px;padding:32px}h1{font-size:clamp(28px,4vw,44px);font-weight:650;letter-spacing:-.04em;margin:12px 0}
p{color:#a9b6c8;line-height:1.6;max-width:850px}.eyebrow{color:#7bdbc0;font-size:12px;letter-spacing:.17em;text-transform:uppercase}
.panel{background:#171e2b;border:1px solid #2b3547;border-radius:16px;padding:20px;margin:20px 0}
.controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,button{background:#202b3d;color:#e7edf4;border:1px solid #3a4962;border-radius:8px;padding:9px 12px;font:inherit}
button{cursor:pointer}input{flex:1;min-width:140px;accent-color:#7bdbc0}.views{display:grid;grid-template-columns:1fr 1fr;gap:20px}
canvas{width:100%;height:auto;display:block;background:#121925;border-radius:10px}.small{font-size:12px;color:#9aa9bc}
.metrics{display:flex;gap:24px;flex-wrap:wrap;margin-top:18px}.metrics b{display:block;font-size:22px;color:#7bdbc0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px;border-bottom:1px solid #2b3547;text-align:left}
.badge{display:inline-block;background:#25394a;color:#b7dfeb;padding:5px 9px;border-radius:6px;font-size:12px}
@media(max-width:700px){body{padding:18px}.views{grid-template-columns:1fr}.panel{padding:14px}}
</style>
<div class="eyebrow">BrainBoxSol / adaptive material memory</div>
<h1>Experience changes the medium.</h1>
<p>Signals fade. The connections they helped train remain. This replay shows actual recorded activity in an 81-node medium, before and after supervised learning.</p>
<span class="badge">__MODE__ · recorded simulation · no browser training</span>
<div class="panel controls"><label>Association <select id="mapping">__MAP_OPTIONS__</select></label>
<label>Cue <select id="cue"><option value="0">A</option><option value="1">B</option></select></label>
<label>Material <select id="arm"><option>Learned</option><option>Before</option><option>Frozen</option><option>Erased</option></select></label>
<button id="play">Play</button><input id="time" aria-label="Simulation time" type="range" min="0" max="40" value="0"><span id="clock">0.0</span></div>
<div class="views"><section class="panel"><h2>Persistent structure</h2><canvas id="structure" width="500" height="500"></canvas>
<p class="small">272 fixed candidate edges. Brighter, thicker lines have higher conductance. Edges below 0.16 are faint; no edges are added or deleted.</p></section>
<section class="panel"><h2>Fast activity</h2><canvas id="activity" width="500" height="500"></canvas>
<p class="small">Each episode begins at zero activity. Sensory input enters on the left; the two fixed motor regions are on the right. Colour uses the same scale across every episode.</p></section></div>
<div class="panel"><div id="status"></div><div class="metrics"><div><span class="small">Upper motor energy</span><b id="upper"></b></div>
<div><span class="small">Lower motor energy</span><b id="lower"></b></div><div><span class="small">Retained budget</span><b id="budget"></b></div></div>
<p class="small">Motor energies are the complete recorded observation window, t = 11.2–16.0. Scrubbing shows activity at the selected time. Before, Frozen and Erased share the original material by design.</p></div>
<div class="panel"><h2>All-seed measurements</h2><table><thead><tr><th>Arm</th><th>Mean accuracy</th></tr></thead><tbody>__ROWS__</tbody></table>
<p>This is a supervised two-cue association task. A six-parameter linear classifier is a strong, cheaper reference. The changing material is a memory mechanism; this experiment does not establish a complete brain or superior digital AI.</p>
<p class="small">Protocol code commit: __COMMIT__</p></div>
<noscript><p>JavaScript is disabled. The measurements above remain readable; enable scripting to play the recorded activity.</p></noscript>
<script type="application/json" id="data">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
let frame=0,playing=false,last=0; const maxU=Math.max(...D.episodes.map(e=>Math.max(...e.states.flat())));
function episode(){return D.episodes.find(e=>e.mapping.join(',')===$('mapping').value&&e.cue===+$('cue').value&&e.arm===$('arm').value)}
function point(i){return [50+(i%D.size)*50,50+Math.floor(i/D.size)*50]}
function draw(){const e=episode();if(!e){$('status').textContent='No recorded episode for this selection.';return;}frame=Math.min(frame,e.states.length-1);$('time').max=e.states.length-1;$('time').value=frame;$('clock').textContent='t = '+e.times[frame].toFixed(1);
for(const kind of ['structure','activity']){const c=$(kind).getContext('2d');c.clearRect(0,0,500,500);
D.edges.forEach(([a,b],i)=>{const g=e.material[i],[x,y]=point(a),[xx,yy]=point(b);c.beginPath();c.moveTo(x,y);c.lineTo(xx,yy);c.lineWidth=kind==='structure'?.4+7*g:1;
c.strokeStyle=kind==='structure'?(g>=.16?'rgba(123,219,192,.85)':'rgba(114,140,170,.15)'):'rgba(80,101,130,.20)';c.stroke()});
for(let i=0;i<D.size*D.size;i++){const [x,y]=point(i),v=e.states[frame][i],h=Math.sqrt(Math.max(v,0)/(maxU||1));c.beginPath();c.arc(x,y,kind==='activity'?4+12*h:3,0,Math.PI*2);c.fillStyle=kind==='activity'?`rgb(${Math.round(35+215*h)},${Math.round(65+145*h)},${Math.round(92-30*h)})`:'#7892aa';c.fill();}
for(const [rows,col,label] of [[[1,2,3],7,'M0'],[[5,6,7],7,'M1']]){c.strokeStyle='#7bdbc0';c.strokeRect(385,rows[0]*50+33,30,134);c.fillStyle='#bde9df';c.font='12px system-ui';c.fillText(label,426,rows[1]*50+55)}
c.fillStyle='#a9b6c8';c.font='12px system-ui';c.fillText('A',17,153);c.fillText('B',17,353);}
$('upper').textContent=e.energies[0].toExponential(3);$('lower').textContent=e.energies[1].toExponential(3);$('budget').textContent=e.material.reduce((a,b)=>a+b,0).toFixed(2);
const correct=e.action===e.target_motor&&e.valid;$('status').textContent=`Seed ${e.seed} · cue ${e.cue?'B':'A'} · ${e.arm} · expected motor ${e.target_motor} · measured ${e.action===null?'no decision':e.action} · ${correct?'correct':'incorrect'}`;
}
for(const id of ['mapping','cue','arm'])$(id).addEventListener('change',()=>{frame=0;draw()});
$('time').addEventListener('input',()=>{frame=+$('time').value;draw()});$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'Pause':'Play'};
function animate(t){const e=episode();if(playing&&e&&t-last>100){frame=(frame+1)%e.states.length;draw();last=t}requestAnimationFrame(animate)}
draw();requestAnimationFrame(animate);
</script></html>'''
