/*
  Live WebSocket server for eeg_act_analyzer outputs.
  - Serves p5 pages
  - Tails NDJSON (window_result, quality_event, session_meta) and broadcasts
  - Optionally tails CSV for raw EEG samples and broadcasts in batches

  Usage:
    PORT=8080 JSON_DIR=../data/json CSV_PATH=../data/eeg_massimo1.csv node server.js

  Defaults (relative to this file):
    JSON_DIR: ../data/json
    CSV_PATH: (unset -> eeg stream disabled)
    PORT: 8080
    WS_PATH: /ws
*/

const fs = require('fs');
const path = require('path');
const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const chokidar = require('chokidar');

const PORT = parseInt(process.env.PORT || getArg('--port', '8080'), 10);
const WS_PATH = process.env.WS_PATH || getArg('--ws_path', '/ws');
const JSON_DIR = path.resolve(__dirname, process.env.JSON_DIR || getArg('--json_dir', '../data/json'));
const CSV_PATH = process.env.CSV_PATH || getArg('--csv_path', '');

function getArg(name, def) {
  const idx = process.argv.findIndex(a => a.startsWith(name + '='));
  if (idx >= 0) return process.argv[idx].split('=')[1];
  return def;
}

const app = express();

// Serve the existing p5js folder and any assets
const p5Path = path.resolve(__dirname, '../p5js');
app.use('/p5js', express.static(p5Path));

// Health and root redirects
app.get('/healthz', (_req, res) => res.send('ok'));
app.get('/', (_req, res) => res.redirect('/p5js/live.html'));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server, path: WS_PATH });

function broadcast(obj) {
  const payload = typeof obj === 'string' ? obj : JSON.stringify(obj);
  wss.clients.forEach(ws => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(payload);
    }
  });
}

wss.on('connection', (ws) => {
  ws.send(JSON.stringify({ type: 'hello', msg: 'connected' }));
});

server.listen(PORT, () => {
  console.log(`[web] listening on http://localhost:${PORT}  (ws path ${WS_PATH})`);
  console.log(`[web] serving p5 from ${p5Path}`);
  console.log(`[web] NDJSON dir: ${JSON_DIR}`);
  if (CSV_PATH) console.log(`[web] CSV path: ${CSV_PATH}`);
});

// ---------------- NDJSON tailer with rotation ----------------
let ndjsonTail = null;
startNdjsonTail(JSON_DIR);

function startNdjsonTail(dir) {
  ensureDir(dir);
  let currentFile = pickLatestSession(dir);
  if (currentFile) {
    console.log(`[ndjson] initial session file: ${currentFile}`);
    ndjsonTail = tailFile(path.join(dir, currentFile), onNdjsonLine, {
      name: 'ndjson',
    });
  } else {
    console.log('[ndjson] no session_*.ndjson found yet, waiting...');
  }

  // Watch for new session files
  const watcher = chokidar.watch(dir, { ignoreInitial: true, depth: 0 });
  watcher.on('add', (filePath) => {
    const bn = path.basename(filePath);
    if (bn.startsWith('session_') && bn.endsWith('.ndjson')) {
      // Pick the latest, may switch
      const latest = pickLatestSession(dir);
      if (latest && (!ndjsonTail || path.basename(ndjsonTail.filePath) !== latest)) {
        console.log(`[ndjson] switching to new session file: ${latest}`);
        if (ndjsonTail) ndjsonTail.close();
        ndjsonTail = tailFile(path.join(dir, latest), onNdjsonLine, { name: 'ndjson' });
      }
    }
  });
}

function pickLatestSession(dir) {
  try {
    const entries = fs.readdirSync(dir).filter(f => f.startsWith('session_') && f.endsWith('.ndjson'));
    if (entries.length === 0) return null;
    // Choose by mtime desc
    const sorted = entries.map(f => {
      const st = fs.statSync(path.join(dir, f));
      return { f, mtime: st.mtimeMs };
    }).sort((a,b) => b.mtime - a.mtime);
    return sorted[0].f;
  } catch (e) { return null; }
}

function onNdjsonLine(line) {
  // Broadcast raw line to clients; it's already JSON
  // Optionally validate: try { JSON.parse(line) } catch {}
  broadcast(line);
}

// ---------------- CSV tailer -> eeg batches ----------------
if (CSV_PATH) {
  tailCsv(CSV_PATH, onCsvSamples, { name: 'csv' });
}

const eegBatch = [];
const BATCH_SIZE = 64; // samples per WS frame
const FLUSH_MS = 100; // flush cadence
let t0 = null;
let announcedT0 = false;
setInterval(() => {
  if (eegBatch.length > 0) {
    broadcast({ type: 'eeg', samples: eegBatch.splice(0, eegBatch.length) });
  }
}, FLUSH_MS);

function onCsvSamples(rows) {
  for (const r of rows) {
    // expected columns: timestamp,TP9,AF7,AF8,TP10,horseshoe_...,blink,jaw
    const ts = parseFloat(r[0]);
    if (!isFinite(ts)) continue;
    if (t0 === null) {
      t0 = ts;
      if (!announcedT0) {
        broadcast({ type: 'eeg_meta', t0_epoch: t0 });
        announcedT0 = true;
      }
    }
    const sample = {
      t: ts - t0,
      TP9: parseFloat(r[1]),
      AF7: parseFloat(r[2]),
      AF8: parseFloat(r[3]),
      TP10: parseFloat(r[4])
    };
    eegBatch.push(sample);
    if (eegBatch.length >= BATCH_SIZE) {
      broadcast({ type: 'eeg', samples: eegBatch.splice(0, eegBatch.length) });
    }
  }
}

// --------------- generic tailers -----------------
function ensureDir(dir) {
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

function tailFile(filePath, onLine, opts={}) {
  const name = opts.name || path.basename(filePath);
  let lastSize = 0;
  let buffer = '';

  function readAppended() {
    fs.stat(filePath, (err, st) => {
      if (err) return; // file may not exist yet
      if (st.size < lastSize) {
        // rotated or truncated
        lastSize = 0;
      }
      if (st.size > lastSize) {
        const stream = fs.createReadStream(filePath, { encoding: 'utf8', start: lastSize, end: st.size - 1 });
        stream.on('data', chunk => {
          buffer += chunk;
          let idx;
          while ((idx = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, idx).trim();
            buffer = buffer.slice(idx + 1);
            if (line.length) onLine(line);
          }
        });
        stream.on('end', () => { lastSize = st.size; });
      }
    });
  }

  // Seed lastSize to end of current file so we only stream new data
  try { lastSize = fs.statSync(filePath).size; } catch {}

  const watcher = chokidar.watch(filePath, { persistent: true, ignoreInitial: true });
  watcher.on('change', readAppended);
  watcher.on('error', err => console.error(`[${name}] watcher error`, err));

  console.log(`[${name}] tailing ${filePath}`);

  return {
    filePath,
    close: () => watcher.close(),
  };
}

function tailCsv(filePath, onRows, opts={}) {
  const name = opts.name || 'csv';
  let lastSize = 0;
  let partial = '';
  let headerParsed = false;

  function readAppended() {
    fs.stat(filePath, (err, st) => {
      if (err) return;
      if (st.size < lastSize) lastSize = 0;
      if (st.size > lastSize) {
        const stream = fs.createReadStream(filePath, { encoding: 'utf8', start: lastSize, end: st.size - 1 });
        stream.on('data', chunk => {
          partial += chunk;
          let idx;
          const rows = [];
          while ((idx = partial.indexOf('\n')) >= 0) {
            const line = partial.slice(0, idx).trim();
            partial = partial.slice(idx + 1);
            if (!line) continue;
            if (!headerParsed) { headerParsed = true; continue; }
            const cols = parseCsvLine(line);
            rows.push(cols);
          }
          if (rows.length) onRows(rows);
        });
        stream.on('end', () => { lastSize = st.size; });
      }
    });
  }

  try { lastSize = fs.statSync(filePath).size; } catch {}

  const watcher = chokidar.watch(filePath, { persistent: true, ignoreInitial: false });
  watcher.on('add', readAppended);
  watcher.on('change', readAppended);
  watcher.on('error', err => console.error(`[${name}] watcher error`, err));

  console.log(`[${name}] tailing ${filePath}`);
}

function parseCsvLine(line) {
  // Simple CSV parser (no quoted commas in our format)
  return line.split(',');
}
