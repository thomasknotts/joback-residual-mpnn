/* ── JR-MPNN Frontend Script ────────────────────────────────────────────────── */

/* ── Tab navigation ─────────────────────────────────────────────────────────── */
document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.remove('hidden');
  });
});

/* ── SMILES example buttons ─────────────────────────────────────────────────── */
document.querySelectorAll('.ex-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.getElementById('smiles-input').value = btn.dataset.smiles;
    triggerVisualize();
  });
});

/* ── Helpers ─────────────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

function showState(state) {
  ['idle-state','spinner','error-box','results-all','result-single'].forEach(id => {
    const el = $(id);
    if (el) el.classList.add('hidden');
  });
  if (state && $(state)) $(state).classList.remove('hidden');
}

function showError(msg) {
  showState('error-box');
  $('error-msg').textContent = msg;
}

function fmtNum(n, dp=3) {
  if (n === undefined || n === null) return '—';
  return Number(n).toFixed(dp);
}

/* ── Visualize ───────────────────────────────────────────────────────────────── */
async function triggerVisualize() {
  const smiles = $('smiles-input').value.trim();
  if (!smiles) { $('mol-card').classList.add('hidden'); return; }

  const res = await fetch('/visualize', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({smiles})
  });
  const data = await res.json();

  if (!data.success) { $('mol-card').classList.add('hidden'); return; }

  $('mol-svg-wrap').innerHTML = data.svg;
  const badges = $('mol-badges');
  badges.innerHTML = `
    <span class="mol-badge">${data.info.formula}</span>
    <span class="mol-badge">${data.info.atoms} atoms</span>
    <span class="mol-badge">${data.info.bonds} bonds</span>
    ${data.info.rings ? `<span class="mol-badge">${data.info.rings} rings</span>` : ''}
  `;
  $('mol-card').classList.remove('hidden');
}

$('viz-btn').addEventListener('click', triggerVisualize);
$('smiles-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') { triggerVisualize(); }
});

/* ── Build property card HTML ────────────────────────────────────────────────── */
function buildPropCard(key, r, delay=0) {
  if (r.error) {
    return `<div class="prop-card error-card" style="animation-delay:${delay}ms">
      <div class="prop-card-header">
        <div>
          <div class="prop-name">${key}</div>
          <div class="prop-fullname">${r.full_name || ''}</div>
        </div>
      </div>
      <div class="prop-value">Error</div>
      <div class="prop-meta" style="color:var(--red);font-size:11px">${r.error}</div>
    </div>`;
  }

  // Build CI bar: map prediction, lower, upper into 0–100%
  const range = r.upper - r.lower;
  const pad   = range * 0.3;
  const barMin = r.lower - pad;
  const barMax = r.upper + pad;
  const barRange = barMax - barMin;
  const fillLeft   = ((r.lower - barMin) / barRange * 100).toFixed(1);
  const fillRight  = (100 - (r.upper - barMin) / barRange * 100).toFixed(1);
  const markerLeft = ((r.prediction - barMin) / barRange * 100).toFixed(1);

  const res = r.residual;
  const resClass = res >= 0 ? 'residual-pos' : 'residual-neg';
  const resSign  = res >= 0 ? '+' : '';

  return `
  <div class="prop-card" style="animation-delay:${delay}ms">
    <div class="prop-card-header">
      <div>
        <div class="prop-name">${key}</div>
        <div class="prop-fullname">${r.full_name}</div>
      </div>
    </div>
    <div class="prop-value">${fmtNum(r.prediction)} <span>${r.unit}</span></div>
    <div class="ci-bar-wrap">
      <div class="ci-label">
        <span>${fmtNum(r.lower)}</span>
        <span>95% CI</span>
        <span>${fmtNum(r.upper)}</span>
      </div>
      <div class="ci-bar">
        <div class="ci-fill" style="left:${fillLeft}%;right:${fillRight}%"></div>
        <div class="ci-marker" style="left:calc(${markerLeft}% - 1px)"></div>
      </div>
    </div>
    <div class="prop-meta">
      <span class="meta-joback">Joback: <span>${fmtNum(r.joback)}</span></span>
      <span class="meta-residual">Δ: <span class="${resClass}">${resSign}${fmtNum(res, 2)}</span></span>
    </div>
  </div>`;
}

/* ── Predict All ─────────────────────────────────────────────────────────────── */
$('predict-all-btn').addEventListener('click', async () => {
  const smiles = $('smiles-input').value.trim();
  if (!smiles) { showError('Please enter a SMILES string.'); return; }

  showState('spinner');
  await triggerVisualize();

  try {
    const res = await fetch('/predict_all', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({smiles})
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); return; }

    const container = $('results-all');
    const ORDER = ['Tm', 'Tb', 'Tc', 'Pc', 'Vc'];
    container.innerHTML = ORDER.map((k, i) =>
      buildPropCard(k, data.results[k] || {error: 'not available'}, i * 60)
    ).join('');
    showState('results-all');
  } catch(e) {
    showError(`Network error: ${e.message}`);
  }
});

/* ── Predict Single ──────────────────────────────────────────────────────────── */
$('predict-one-btn').addEventListener('click', async () => {
  const smiles = $('smiles-input').value.trim();
  const prop   = $('prop-select').value;
  if (!smiles) { showError('Please enter a SMILES string.'); return; }
  if (!prop)   { showError('Please select a property.'); return; }

  showState('spinner');
  await triggerVisualize();

  try {
    const res = await fetch('/predict_all', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({smiles})
    });
    const data = await res.json();
    if (!data.success) { showError(data.error); return; }

    const container = $('result-single');
    container.innerHTML = buildPropCard(prop, data.results[prop] || {error: 'not available'});
    showState('result-single');
  } catch(e) {
    showError(`Network error: ${e.message}`);
  }
});

/* ── Batch prediction ────────────────────────────────────────────────────────── */
let batchResults = null;

// File drag-drop
const fileDrop = $('file-drop');
fileDrop.addEventListener('dragover', e => { e.preventDefault(); fileDrop.classList.add('dragover'); });
fileDrop.addEventListener('dragleave', () => fileDrop.classList.remove('dragover'));
fileDrop.addEventListener('drop', e => {
  e.preventDefault(); fileDrop.classList.remove('dragover');
  const f = e.dataTransfer.files[0];
  if (f) { $('batch-file').files = e.dataTransfer.files; $('file-label').textContent = f.name; }
});
$('batch-file').addEventListener('change', e => {
  if (e.target.files[0]) $('file-label').textContent = e.target.files[0].name;
});

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return null;
  const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
  const smilesIdx = headers.findIndex(h => h === 'smiles');
  const nameIdx   = headers.findIndex(h => h === 'name');
  if (smilesIdx === -1) return null;
  return lines.slice(1).map(line => {
    const cols = line.split(',').map(c => c.trim().replace(/^"|"$/g, ''));
    return {
      smiles: cols[smilesIdx] || '',
      name:   nameIdx !== -1 ? cols[nameIdx] : cols[smilesIdx]?.slice(0, 24),
    };
  }).filter(r => r.smiles);
}

$('batch-run-btn').addEventListener('click', async () => {
  const file = $('batch-file').files[0];
  const prop = $('batch-prop-select').value;

  if (!file) { alert('Please choose a CSV file.'); return; }
  if (!prop) { alert('Please select a property.'); return; }

  const text = await file.text();
  const rows = parseCSV(text);
  if (!rows || rows.length === 0) {
    alert('CSV must have a SMILES column. No valid rows found.');
    return;
  }
  if (rows.length > 500) { alert('Maximum 500 molecules per batch.'); return; }

  $('batch-run-btn').disabled = true;
  $('batch-run-btn').textContent = 'Running…';

  try {
    const res = await fetch('/batch', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({smiles_list: rows, property: prop})
    });
    const data = await res.json();
    if (!data.success) { alert(data.error); return; }

    batchResults = data;
    renderBatchResults(data);
  } catch(e) {
    alert(`Error: ${e.message}`);
  } finally {
    $('batch-run-btn').disabled = false;
    $('batch-run-btn').textContent = 'Run Batch';
  }
});

function renderBatchResults(data) {
  const success = data.results.filter(r => !r.error).length;
  const failed  = data.results.filter(r =>  r.error).length;

  $('batch-stats').innerHTML =
    `<span>${data.results.length}</span> total &nbsp;·&nbsp;
     <span style="color:var(--green)">${success}</span> OK &nbsp;·&nbsp;
     <span style="color:var(--red)">${failed}</span> failed`;

  const tbody = $('batch-tbody');
  tbody.innerHTML = data.results.map(r => {
    if (r.error) {
      return `<tr>
        <td>${r.name || '—'}</td>
        <td class="smiles-cell" title="${r.smiles}">${r.smiles}</td>
        <td>—</td><td>—</td><td>—</td><td>—</td>
        <td class="status-err">✗ ${r.error}</td>
      </tr>`;
    }
    return `<tr>
      <td>${r.name || '—'}</td>
      <td class="smiles-cell" title="${r.smiles}">${r.smiles}</td>
      <td>${fmtNum(r.prediction)} ${r.unit}</td>
      <td>${fmtNum(r.lower)}</td>
      <td>${fmtNum(r.upper)}</td>
      <td>${fmtNum(r.joback)}</td>
      <td class="status-ok">✓</td>
    </tr>`;
  }).join('');

  $('batch-results').classList.remove('hidden');
  $('batch-download-btn').classList.remove('hidden');
}

/* ── Batch CSV download ──────────────────────────────────────────────────────── */
$('batch-download-btn').addEventListener('click', () => {
  if (!batchResults) return;
  const prop = batchResults.property;
  const unit = batchResults.unit;
  const header = `Name,SMILES,${prop} (${unit}),Lower CI (${unit}),Upper CI (${unit}),Joback (${unit}),Status\n`;
  const rows = batchResults.results.map(r => {
    const esc = v => `"${String(v).replace(/"/g,'""')}"`;
    if (r.error) return [esc(r.name||''), esc(r.smiles), '','','','', esc(`Error: ${r.error}`)].join(',');
    return [esc(r.name||''), esc(r.smiles),
            r.prediction, r.lower, r.upper, r.joback, 'OK'].join(',');
  }).join('\n');
  const blob = new Blob([header + rows], {type:'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `jrmpnn_${prop.toLowerCase()}_predictions.csv`;
  a.click();
});

$('batch-clear-btn').addEventListener('click', () => {
  batchResults = null;
  $('batch-tbody').innerHTML = '';
  $('batch-results').classList.add('hidden');
  $('batch-download-btn').classList.add('hidden');
  $('batch-file').value = '';
  $('file-label').textContent = 'Choose file or drag & drop';
  $('batch-prop-select').value = '';
  $('batch-stats').innerHTML = '';
});

/* ── Initial state ───────────────────────────────────────────────────────────── */
showState('idle-state');
