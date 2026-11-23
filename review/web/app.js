/* Browser-local, no telemetry and no automatic upload. */
'use strict';
const $ = id => document.getElementById(id);
const Core = window.ReviewCore;
let pkg = null, ratings = {}, index = 0, persistOK = true;
const descriptions = {
  accuracy: ['Accuracy', 'Matches the visible events without invented details.'],
  usefulness: ['Usefulness', 'Captures important visual information.'],
  non_redundancy: ['Non-redundancy', 'Adds information beyond dialogue and obvious sounds.'],
  clarity_timing: ['Clarity and timing', 'Clear, concise, and appropriately timed.']
};
function notice(text) { $('notice').textContent = text; }
function key() { return `pyro-review-v1:${pkg.study_id}:${pkg.package_id}:${pkg.reviewer}`; }
function snapshot() { return {schema_version:1, study_id:pkg.study_id, package_id:pkg.package_id, reviewer:pkg.reviewer, demo:pkg.demo, exported_at:new Date().toISOString(), ratings:Object.values(ratings)}; }
function persist() {
  try { localStorage.setItem(key(),JSON.stringify(snapshot())); persistOK=true; }
  catch (_) { persistOK=false; notice('Browser storage is unavailable or full. Export your reviews before closing this page.'); }
}
function setPackage(p) {
  Core.validatePackage(p);
  if (pkg) capture(false);
  document.querySelectorAll('video').forEach(v=>v.pause());
  pkg=p; ratings={}; index=0; persistOK=true;
  try {
    const stored=localStorage.getItem(key());
    if (stored) Core.validateImport(JSON.parse(stored),pkg).forEach(r=>ratings[r.item_id]=r);
  } catch (_) { notice('Saved data could not be read. Restore a downloaded export, or continue with a new draft.'); }
  $('welcome').hidden=true; $('review').hidden=false; $('export').disabled=false;
  $('identity').textContent=`${pkg.reviewer} · ${pkg.items.length} comparisons`;
  if (pkg.demo) notice('PRACTICE ONLY — Synthetic footage. These ratings are excluded from the real study.');
  else notice('Your assignment is loaded. Save each completed review, then export and send the file to your coordinator.');
  render();
}
function renderScore(version) {
  const container=$(`scores-${version.toLowerCase()}`); container.replaceChildren();
  for (const c of Core.criteria) {
    const field=document.createElement('fieldset'); field.className='criterion';
    const legend=document.createElement('legend'); legend.textContent=descriptions[c][0]; field.append(legend);
    const help=document.createElement('p'); help.textContent=descriptions[c][1]; field.append(help);
    const scale=document.createElement('div'); scale.className='scale';
    for (let n=1;n<=5;n++) {
      const label=document.createElement('label'), radio=document.createElement('input'), span=document.createElement('span');
      radio.type='radio'; radio.name=`${version}-${c}`; radio.value=n;
      radio.setAttribute('aria-label',`${version}: ${descriptions[c][0]}, ${n} out of 5`);
      span.textContent=n; label.append(radio,span); scale.append(label);
    }
    field.append(scale); container.append(field);
  }
  for (const c of Core.collisions) {
    const label=document.createElement('label'); label.className='collision';
    label.textContent=c==='dialogue_overlap'?'Overlaps original dialogue?':'Narrations overlap each other?';
    const select=document.createElement('select'); select.id=`${version}-${c}`;
    for (const value of ['', 'Yes','No','Uncertain']) { const o=document.createElement('option'); o.value=value; o.textContent=value||'Choose an answer'; select.append(o); }
    label.append(select); container.append(label);
  }
}
function progress() {
  const done=pkg.items.filter(i=>ratings[i.id]?.completed).length;
  $('progress').max=pkg.items.length; $('progress').value=done;
  $('progress-text').textContent=`${done} of ${pkg.items.length} completed`;
  $('progress-percent').textContent=`${Math.round(done/pkg.items.length*100)}%`;
  $('items').replaceChildren();
  pkg.items.forEach((item,n)=>{
    const b=document.createElement('button'); b.textContent=String(n+1).padStart(2,'0');
    b.className=(n===index?'active ':'')+(ratings[item.id]?.completed?'complete':'');
    b.setAttribute('aria-label',`Comparison ${n+1}${ratings[item.id]?.completed?', completed':''}`);
    if(n===index)b.setAttribute('aria-current','step');
    b.onclick=()=>navigate(n); $('items').append(b);
  });
  $('saved-state').textContent=ratings[pkg.items[index].id]?.completed?'Completed':ratings[pkg.items[index].id]?(persistOK?'Draft saved':'Draft in memory'):'Not reviewed';
}
function restoreFields(r={}) {
  document.querySelectorAll('input[type=radio]').forEach(el=>el.checked=false);
  for (const v of ['X','Y']) {
    for(const c of Core.criteria) {
      const value=r[v]?.[c];
      if(Number.isInteger(value)&&value>=1&&value<=5)document.querySelector(`input[name="${v}-${c}"][value="${value}"]`).checked=true;
    }
    for(const c of Core.collisions) $(v+'-'+c).value=['Yes','No','Uncertain'].includes(r[v]?.[c])?r[v][c]:'';
  }
  if(Core.preferences.includes(r.preference)) [...document.querySelectorAll('input[name=preference]')].find(e=>e.value===r.preference).checked=true;
  $('evidence').value=r.evidence||''; $('issues').value=r.issues||''; $('independent').checked=!!r.independent;
}
function render() {
  const item=pkg.items[index];
  $('item-position').textContent=`${pkg.demo?'PRACTICE · ':''}COMPARISON ${index+1} OF ${pkg.items.length}`;
  $('item-title').textContent=item.id;
  for(const [id,v] of [['original','original'],['video-x','X'],['video-y','Y']]) {
    const player=$(id); player.pause(); player.src=item.media[v]; player.load();
  }
  renderScore('X'); renderScore('Y'); restoreFields(ratings[item.id]);
  $('validation').textContent=''; $('previous').disabled=index===0; $('next').disabled=index===pkg.items.length-1;
  progress();
}
function readFields() {
  const r={item_id:pkg.items[index].id,preference:document.querySelector('input[name=preference]:checked')?.value||'',evidence:$('evidence').value,issues:$('issues').value,independent:$('independent').checked};
  for(const v of ['X','Y']) {
    r[v]={};
    for(const c of Core.criteria) {const input=document.querySelector(`input[name="${v}-${c}"]:checked`); r[v][c]=input?Number(input.value):null;}
    for(const c of Core.collisions) r[v][c]=$(v+'-'+c).value;
  }
  return r;
}
function capture(edited) {
  if(!pkg)return;
  const r=readFields(), old=ratings[r.item_id];
  if(!edited&&!old&&!r.preference&&!r.evidence&&!r.independent&&!r.issues&&!Object.values(r.X).some(Boolean)&&!Object.values(r.Y).some(Boolean))return;
  r.completed=edited?false:!!old?.completed;
  r.updated_at=edited||!old?new Date().toISOString():old.updated_at;
  ratings[r.item_id]=r; persist(); progress();
}
function navigate(n){capture(false); index=n; render(); $('item-title').scrollIntoView({block:'start',behavior:'smooth'});}
function download(name,content){const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([content],{type:'application/json'})); a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('rating-form').onsubmit=e=>{
  e.preventDefault(); const r=readFields(); const error=Core.validateRating(r);
  $('validation').textContent=error; if(error)return;
  r.completed=true; r.updated_at=new Date().toISOString(); ratings[r.item_id]=r; persist(); progress();
  const done=pkg.items.filter(i=>ratings[i.id]?.completed).length;
  notice(done===pkg.items.length?'All comparisons completed. Export your reviews and send the file to the coordinator.':'Review saved. Continue with Next item when you are ready.');
};
$('review').addEventListener('input',e=>{if(e.target.matches('input,textarea,select'))capture(true);});
$('review').addEventListener('change',e=>{if(e.target.matches('select'))capture(true);});
$('previous').onclick=()=>navigate(index-1); $('next').onclick=()=>navigate(index+1);
$('export').onclick=()=>{if(!pkg)return;capture(false);download(`${pkg.demo?'PRACTICE-':''}${pkg.reviewer.replace(/[^a-z0-9_-]/gi,'_')}-reviews.json`,JSON.stringify(snapshot(),null,2));notice('Export downloaded. Send this file to the coordinator; nothing is uploaded automatically.');};
$('load').onclick=()=>$('package-file').click();
$('package-file').onchange=async e=>{try{const f=e.target.files[0];if(f)setPackage(JSON.parse(await f.text()));}catch(err){notice(err.message);}finally{e.target.value='';}};
$('restore').onclick=()=>{if(!pkg){notice('Load your assignment package before restoring an export.');return;}$('restore-file').click();};
$('restore-file').onchange=async e=>{
  try {const f=e.target.files[0];if(!f)return;const imported=Core.validateImport(JSON.parse(await f.text()),pkg);capture(false);
    const conflicts=imported.filter(r=>ratings[r.item_id]&&JSON.stringify(ratings[r.item_id])!==JSON.stringify(r));
    if(conflicts.length&&!confirm(`Replace ${conflicts.length} existing local records with records from this export? Export your current work first if needed.`))return;
    imported.forEach(r=>ratings[r.item_id]=r);persist();render();notice('Export restored. Other local items were retained.');
  } catch(err){notice(err.message);}finally{e.target.value='';}
};
$('demo').onclick=async()=>{try{const r=await fetch('demo/package.json');if(!r.ok)throw Error('Practice package is unavailable. Start the local HTTP server described in the README.');setPackage(await r.json());}catch(err){notice(err.message);}};
document.querySelectorAll('video').forEach(player=>{
  player.addEventListener('play',()=>document.querySelectorAll('video').forEach(other=>{if(other!==player)other.pause();}));
  player.addEventListener('error',()=>{if(pkg)notice('A video could not be loaded. Check that the media folder is present. If it remains unavailable, record Cannot assess and explain the problem.');});
});
window.addEventListener('beforeunload',()=>capture(false));
(async()=>{try{const r=await fetch('review-package.json');if(r.ok)setPackage(await r.json());}catch(_){notice('Open this folder through the local HTTP server in the README to load media and assignments.');}})();
