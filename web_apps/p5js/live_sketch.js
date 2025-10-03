// EEG Live Viewer p5.js sketch
// - Connects to WebSocket provided by web/server.js
// - Renders 4 stacked channel plots (TP9, AF7, AF8, TP10)
// - Overlays chirplet windows from NDJSON window_result messages
// - Shows quality events and session meta
// - Optional audio UI (volume) wired to chirplet_audio.js (no live synthesis yet)

let ws;
let wsStateEl, wsUrlLabelEl;
let session = {
  id: '-', fs: 256, win: 0, hop: 0, order: 0, overlap: 0, backend: '-', dict: 0,
};
let quality = { hs: [ '.', '.', '.', '.' ], blink: 0, jaw: 0 };

const channels = [ 'TP9', 'AF7', 'AF8', 'TP10' ];
const chanColors = {
  TP9: [210, 80, 90],
  AF7: [140, 80, 90],
  AF8: [40, 80, 90],
  TP10:[300, 80, 90],
};

// Ring buffers for live EEG samples (per-channel)
const winSecDefault = 15;
let windowSec = winSecDefault;
let showRaw = true;
let showChirps = true;
let audioOn = false;
let masterVolume = 0.2;

const eegBuffers = {
  TP9: [], AF7: [], AF8: [], TP10: []
};
let lastT = 0; // seconds (relative to server t0)
let t0_epoch = null; // wall-clock seconds from CSV first sample
let sessionStartEpoch = null; // wall-clock seconds from session_meta.session_id
let haveEEG = false;
let chirpTimeOffset = 0; // seconds to add to tc_seconds to align to EEG t-axis
let chirpCalibrated = false;

// Chirplet overlays per-channel: sparse list of {t0, t1, coeff}
const chirpOverlays = {
  TP9: [], AF7: [], AF8: [], TP10: []
};

function setup() {
  const holder = document.getElementById('canvas-container');
  const w = holder ? holder.clientWidth : 1200;
  const h = 720;
  const cnv = createCanvas(w, h);
  cnv.parent('canvas-container');
  try { pixelDensity(1); } catch(e) {}
  try { frameRate(30); } catch(e) {}
  colorMode(HSB, 360, 100, 100, 1.0);

  // UI wiring
  wsStateEl = document.getElementById('wsState');
  wsUrlLabelEl = document.getElementById('wsUrlLabel');
  const url = (window.WS_URL || ((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws'));
  wsUrlLabelEl.textContent = url;

  const showRawEl = document.getElementById('showRaw');
  const showChirpsEl = document.getElementById('showChirps');
  const windowSecEl = document.getElementById('windowSec');
  const windowSecVal = document.getElementById('windowSecVal');
  const audioOnEl = document.getElementById('audioOn');
  const masterVolEl = document.getElementById('masterVolume');
  const masterVolVal = document.getElementById('masterVolumeValue');

  if (showRawEl) showRawEl.addEventListener('change', () => { showRaw = showRawEl.checked; });
  if (showChirpsEl) showChirpsEl.addEventListener('change', () => { showChirps = showChirpsEl.checked; });
  if (windowSecEl) {
    windowSec = Number(windowSecEl.value || winSecDefault);
    windowSecVal.textContent = String(windowSec);
    windowSecEl.addEventListener('input', () => {
      windowSec = Number(windowSecEl.value || winSecDefault);
      windowSecVal.textContent = String(windowSec);
    });
  }
  if (audioOnEl) audioOnEl.addEventListener('change', () => {
    audioOn = audioOnEl.checked;
    if (audioOn) ensureAudioResumed();
  });
  if (masterVolEl && masterVolVal) {
    masterVolVal.textContent = Number(masterVolEl.value).toFixed(2);
    try { if (window.chirpletAudio) chirpletAudio.init(); } catch(e) {}
    setMasterVol(Number(masterVolEl.value));
    chirpletAudio.setLimiterEnabled(true);
    masterVolEl.addEventListener('input', () => {
      masterVolVal.textContent = Number(masterVolEl.value).toFixed(2);
      setMasterVol(Number(masterVolEl.value));
    });
  }

  // Connect WS
  openWS(url);
}

function setMasterVol(v) {
  masterVolume = Math.max(0, Math.min(1, v));
  if (window.chirpletAudio && chirpletAudio.setMasterVolume) {
    try { chirpletAudio.setMasterVolume(masterVolume); } catch(e) {}
  }
}

function ensureAudioResumed() {
  try {
    if (window.chirpletAudio && chirpletAudio.init) chirpletAudio.init();
  } catch(e) {}
  try {
    if (typeof getAudioContext === 'function') {
      const ctx = getAudioContext();
      if (ctx && ctx.state !== 'running' && ctx.resume) ctx.resume();
    }
  } catch(e) {}
}

function windowResized() {
  const holder = document.getElementById('canvas-container');
  const w = holder ? holder.clientWidth : width;
  resizeCanvas(w, height);
}

function draw() {
  background(255);

  // Header status from DOM
  updateStatusDom();

  // Plot area params
  const leftPad = 60;
  const rightPad = 10;
  const topPad = 10;
  const bottomPad = 10;
  const plotW = width - leftPad - rightPad;
  const plotH = height - topPad - bottomPad;
  const rows = 4;
  const rowH = plotH / rows;

  // time window [tStart, tEnd]
  const tEnd = lastT;
  const tStart = Math.max(0, tEnd - windowSec);

  stroke(0, 0, 85); strokeWeight(1);
  // Draw channel plots
  for (let r = 0; r < rows; r++) {
    const ch = channels[r];
    const y0 = topPad + r * rowH;
    const y1 = y0 + rowH;

    // Horizontal grid midline
    stroke(0,0,90); line(leftPad, (y0+y1)/2, leftPad + plotW, (y0+y1)/2);

    // Title
    noStroke(); fill(0); textAlign(LEFT, TOP); textSize(12);
    text(ch, 10, y0 + 6);

    // Compute scale from visible samples
    const buf = eegBuffers[ch];
    let minV = 0, maxV = 0;
    if (buf && buf.length) {
      for (let i = buf.length - 1; i >= 0; i--) {
        const s = buf[i];
        if (s.t < tStart) break;
        if (s.v < minV) minV = s.v;
        if (s.v > maxV) maxV = s.v;
      }
    }
    const amp = Math.max(1e-6, Math.max(Math.abs(minV), Math.abs(maxV)));
    const yScale = (rowH * 0.4) / amp; // 80% height

    // Raw signal polyline (decimate over the visible time window only)
    if (showRaw && buf && buf.length > 1) {
      // find first index within [tStart, tEnd]
      let i0 = 0;
      while (i0 < buf.length && buf[i0].t < tStart) i0++;
      if (i0 < buf.length) {
        stroke(...chanColors[ch], 1); strokeWeight(1.5); noFill();
        beginShape();
        const visibleCount = buf.length - i0;
        const maxVerts = Math.max(100, Math.floor(plotW * 1.5));
        const stepIdx = Math.max(1, Math.floor(visibleCount / maxVerts));
        for (let i = i0; i < buf.length; i += stepIdx) {
          const s = buf[i];
          if (s.t > tEnd) break;
          const x = mapTime(s.t, tStart, tEnd, leftPad, leftPad + plotW);
          const y = (y0 + y1) / 2 - s.v * yScale;
          vertex(x, y);
        }
        // ensure the last visible point is drawn
        const last = buf[buf.length - 1];
        if (last && last.t >= tStart && last.t <= tEnd) {
          const xL = mapTime(last.t, tStart, tEnd, leftPad, leftPad + plotW);
          const yL = (y0 + y1) / 2 - last.v * yScale;
          vertex(xL, yL);
        }
        endShape();
      }
    }

    // Chirplet overlays as translucent boxes
    if (showChirps) {
      const lst = chirpOverlays[ch];
      if (lst && lst.length) {
        for (const c of lst) {
          if (c.t1 < tStart || c.t0 > tEnd) continue;
          const x0 = mapTime(Math.max(c.t0, tStart), tStart, tEnd, leftPad, leftPad + plotW);
          const x1 = mapTime(Math.min(c.t1, tEnd), tStart, tEnd, leftPad, leftPad + plotW);
          const w = Math.max(1, x1 - x0);
          noStroke(); fill(...chanColors[ch], 0.12 + 0.20*Math.min(1, Math.abs(c.coeff))); // alpha by coeff
          rect(x0, y0+4, w, rowH-8);
        }
      }
    }

    // Channel frame
    noFill(); stroke(0,0,60); rect(leftPad, y0, plotW, rowH);
  }
}

function mapTime(t, tStart, tEnd, x0, x1) {
  if (tEnd <= tStart) return x1;
  const f = (t - tStart) / (tEnd - tStart);
  return x0 + Math.max(0, Math.min(1, f)) * (x1 - x0);
}

function updateStatusDom() {
  setText('sessionId', session.id);
  setText('fs', String(session.fs));
  setText('win', String(session.win));
  setText('hop', String(session.hop));
  setText('ord', String(session.order));
  setText('backend', session.backend);
  setText('dict', String(session.dict));
  setText('hs', `[${quality.hs.join(',')}]`);
  setText('blink', String(quality.blink));
  setText('jaw', String(quality.jaw));
}

function setText(id, v) {
  const el = document.getElementById(id);
  if (el) el.textContent = v;
}

function openWS(url) {
  try { if (ws) ws.close(); } catch(e) {}
  ws = new WebSocket(url);
  ws.binaryType = 'arraybuffer';

  function setState(txt, cls) {
    if (wsStateEl) {
      wsStateEl.textContent = txt;
      wsStateEl.className = 'pill ' + (cls || 'warn');
    }
  }

  setState('connecting...', 'warn');

  ws.onopen = () => setState('connected', 'ok');
  ws.onclose = () => setState('disconnected', 'bad');
  ws.onerror = () => setState('error', 'bad');

  ws.onmessage = (ev) => {
    let msg;
    try { msg = typeof ev.data === 'string' ? JSON.parse(ev.data) : null; } catch (e) { return; }
    if (!msg || !msg.type) return;

    switch (msg.type) {
      case 'eeg_meta': {
        const t0e = Number(msg.t0_epoch);
        if (isFinite(t0e) && (t0_epoch === null)) t0_epoch = t0e;
        break;
      }
      case 'session_meta': {
        session.id = msg.session_id || session.id;
        session.fs = Number(msg.fs) || session.fs;
        session.win = Number(msg.window_size) || session.win;
        session.hop = Number(msg.hop) || session.hop;
        session.order = Number(msg.order) || session.order;
        session.overlap = Number(msg.overlap) || session.overlap;
        session.backend = msg.backend || session.backend;
        session.dict = Number(msg.dict_size) || session.dict;
        // Parse session start epoch from session_id (YYYY-MM-DDTHH:MM:SS)
        if (typeof msg.session_id === 'string' && msg.session_id.length >= 19) {
          const d = new Date(msg.session_id);
          const e = d.getTime() / 1000;
          if (isFinite(e)) sessionStartEpoch = e;
        }
        break;
      }
      case 'quality_event': {
        if (Array.isArray(msg.horseshoe) && msg.horseshoe.length === 4) quality.hs = msg.horseshoe.slice();
        quality.blink = Number(msg.blink) || 0;
        quality.jaw = Number(msg.jaw_clench) || 0;
        break;
      }
      case 'window_result': {
        const ch = msg.channel;
        if (!channels.includes(ch)) break;
        const used = Number(msg.used_order) || 0;
        const arr = chirpOverlays[ch];
        // Calibrate chirp time offset once we have EEG and session meta
        if (!chirpCalibrated && haveEEG && isFinite(Number(session.fs)) && isFinite(Number(session.win)) && isFinite(Number(msg.window_start))) {
          const fs = Number(session.fs);
          const win = Number(session.win);
          const ws = Number(msg.window_start);
          const windowEndRel = (ws + win) / fs; // seconds relative to sample0
          // Align so that window end maps near our current EEG time
          chirpTimeOffset = lastT - windowEndRel;
          chirpCalibrated = true;
        }
        // append chirps
        if (Array.isArray(msg.chirplets)) {
          // Schedule audio playback if enabled
          if (audioOn && window.chirpletAudio && typeof chirpletAudio.playLiveChirplets === 'function') {
            try {
              // Map to live chirp objects
              const chirps = msg.chirplets.map(c => ({
                tc_seconds: Number(c.tc_seconds),
                duration_ms: Number(c.duration_ms),
                fc_hz: Number(c.fc_hz),
                c_hz_per_s: Number(c.c_hz_per_s),
                coeff: Number(c.coeff),
              })).filter(c => isFinite(c.tc_seconds) && isFinite(c.duration_ms) && c.duration_ms > 0);
              if (chirps.length) chirpletAudio.playLiveChirplets(chirps, Number(session.fs) || 256);
            } catch(e) { /* ignore */ }
          }
          for (const c of msg.chirplets) {
            let tc = Number(c.tc_seconds);
            const dur = Number(c.duration_ms) / 1000.0;
            if (!isFinite(tc) || !isFinite(dur)) continue;
            // Align tc_seconds to EEG-relative time
            if (chirpCalibrated) {
              tc = tc + chirpTimeOffset;
            } else if (t0_epoch !== null && sessionStartEpoch !== null) {
              // Fallback: epoch-based alignment if calibration isn't available yet
              tc = (sessionStartEpoch + tc) - t0_epoch;
            }
            const t0 = tc;
            const t1 = tc + Math.max(0, dur);
            const coeff = Number(c.coeff) || 0;
            arr.push({ t0, t1, coeff });
            if (!haveEEG && t1 > lastT) lastT = t1; // only advance without EEG stream
          }
        }
        // prune old ones
        pruneChirps(arr);
        break;
      }
      case 'eeg': {
        if (!Array.isArray(msg.samples)) break;
        for (const s of msg.samples) {
          const t = Number(s.t);
          if (!isFinite(t)) continue;
          haveEEG = true;
          lastT = Math.max(lastT, t);
          pushSample('TP9', t, s.TP9);
          pushSample('AF7', t, s.AF7);
          pushSample('AF8', t, s.AF8);
          pushSample('TP10', t, s.TP10);
        }
        // prune buffers
        pruneBuffers();
        break;
      }
      default:
        break;
    }
  };
}

function pushSample(ch, t, v) {
  const buf = eegBuffers[ch];
  if (!isFinite(v)) return;
  buf.push({ t, v: Number(v) });
}

function pruneBuffers() {
  const t0 = Math.max(0, lastT - windowSec);
  for (const ch of channels) {
    const buf = eegBuffers[ch];
    // Drop older than t0; leave a little margin
    while (buf.length > 0 && buf[0].t < t0 - 1) buf.shift();
  }
}

function pruneChirps(arr) {
  const t0 = Math.max(0, lastT - windowSec);
  let i = 0;
  while (i < arr.length && arr[i].t1 < t0 - 2) i++;
  if (i > 0) arr.splice(0, i);
}
