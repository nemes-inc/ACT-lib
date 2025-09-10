let analysisData = null;
let csvData = null;
let canvas;
let zoomLevel = 1;
let panX = 0;
let showOriginal = true;
let showChirplets = true;
let chirpletVisibility = [];
let signalBuffer = [];
let chirpletBuffers = [];
let overCanvas = false; // track whether mouse is over the canvas
let reconstructionBuffer = [];
let residualBuffer = [];
let scaleFactorSignal = 1; // cached y-scale factor for drawing
let showReconstruction = false;
let showResidual = false;
let showFInst = false;
let signalEnergy = 0;
let reconstructionEnergy = 0;
let residualEnergy = 0;

function setup() {
    const holder = document.getElementById('canvas-container');
    const w = holder ? holder.clientWidth : 1000;
    canvas = createCanvas(w, 600);
    canvas.parent('canvas-container');
    // Enforce visible size in case external CSS interferes
    try {
        canvas.style('display', 'block');
        canvas.style('width', '100%');
        canvas.style('height', '600px');
    } catch(e) {}

    // Performance tuning: lower device pixel density and FPS to reduce per-frame work
    try { pixelDensity(1); } catch(e) {}
    try { frameRate(30); } catch(e) {}

    colorMode(HSB, 360, 100, 100);

    // Track pointer entering/leaving the canvas to gate drag panning
    try {
        canvas.mouseOver(() => { overCanvas = true; });
        canvas.mouseOut(() => { overCanvas = false; });
    } catch(e) {}

    // Set up event listeners
    document.getElementById('zoom').addEventListener('input', updateZoom);
    document.getElementById('panX').addEventListener('input', updatePan);
    document.getElementById('showOriginal').addEventListener('change', updateShowOriginal);
    const sc = document.getElementById('showChirplets');
    if (sc) sc.addEventListener('change', updateShowChirplets);
    const sr = document.getElementById('showReconstruction');
    if (sr) sr.addEventListener('change', updateShowReconstruction);
    const sres = document.getElementById('showResidual');
    if (sres) sres.addEventListener('change', updateShowResidual);
    const sfi = document.getElementById('showFInst');
    if (sfi) sfi.addEventListener('change', updateShowFInst);

    // Audio controls wiring
    if (window.chirpletAudio) {
        console.log('Chirplet Audio initializing...');
        chirpletAudio.init();
        console.log('Chirplet Audio initialized');

        const volEl = document.getElementById('masterVolume');
        const volVal = document.getElementById('masterVolumeValue');
        if (volEl && volVal) {
            volVal.textContent = Number(volEl.value).toFixed(2);
            chirpletAudio.setMasterVolume(parseFloat(volEl.value));
            volEl.addEventListener('input', () => {
                volVal.textContent = Number(volEl.value).toFixed(2);
                chirpletAudio.setMasterVolume(parseFloat(volEl.value));
            });
        }

        const limEl = document.getElementById('limiterToggle');
        if (limEl) {
            limEl.checked = false; // default off
            chirpletAudio.setLimiterEnabled(false);
            limEl.addEventListener('change', () => {
                chirpletAudio.setLimiterEnabled(limEl.checked);
            });
        }

        const pvoEl = document.getElementById('playVisibleOnly');
        if (pvoEl) {
            chirpletAudio.setPlayVisibleOnly(pvoEl.checked);
            pvoEl.addEventListener('change', () => {
                chirpletAudio.setPlayVisibleOnly(pvoEl.checked);
            });
        }

        const psEl = document.getElementById('pitchScale');
        const psVal = document.getElementById('pitchScaleValue');
        if (psEl && psVal) {
            psVal.textContent = `${psEl.value}×`;
            chirpletAudio.setPitchScale(parseFloat(psEl.value));
            psEl.addEventListener('input', () => {
                psVal.textContent = `${psEl.value}×`;
                chirpletAudio.setPitchScale(parseFloat(psEl.value));
            });
        }

        const ckEl = document.getElementById('coverageK');
        const ckVal = document.getElementById('coverageKValue');
        if (ckEl && ckVal) {
            ckVal.textContent = Number(ckEl.value).toFixed(1);
            chirpletAudio.setCoverageK(parseFloat(ckEl.value));
            ckEl.addEventListener('input', () => {
                ckVal.textContent = Number(ckEl.value).toFixed(1);
                chirpletAudio.setCoverageK(parseFloat(ckEl.value));
            });
        }

        // Export frequency mode (export only)
        const efm = document.getElementById('exportFreqMode');
        if (efm) {
            chirpletAudio.setExportFrequencyMode(efm.value);
            efm.addEventListener('change', () => {
                chirpletAudio.setExportFrequencyMode(efm.value);
            });
        }

        const playBtn = document.getElementById('playBtn');
        const stopBtn = document.getElementById('stopBtn');
        const exportBtn = document.getElementById('exportWavBtn');
        if (playBtn) {
            playBtn.addEventListener('click', async () => {
                if (!analysisData || !csvData) {
                    alert('Load a JSON analysis and CSV first.');
                    return;
                }
                try {
                    chirpletAudio.playFromAnalysis(analysisData, chirpletVisibility || []);
                } catch (e) {
                    console.error('Audio play error:', e);
                }
            });
        }
        if (stopBtn) {
            stopBtn.addEventListener('click', () => {
                try { chirpletAudio.stopAll(); } catch(e) {}
            });
        }
        if (exportBtn) {
            exportBtn.addEventListener('click', async () => {
                if (!analysisData) {
                    alert('Load a JSON analysis first.');
                    return;
                }
                try {
                    await chirpletAudio.exportWav(analysisData, chirpletVisibility || []);
                } catch (e) {
                    console.error('Export WAV error:', e);
                    alert('Export failed. See console for details.');
                }
            });
        }
    }
}

function windowResized() {
    const holder = document.getElementById('canvas-container');
    const w = holder ? holder.clientWidth : width;
    resizeCanvas(w, 600);
    try {
        // Keep style synced
        canvas.style('width', '100%');
        canvas.style('height', '600px');
    } catch(e) {}
}

// Compute visible sample range [start, end) for a buffer of given length under current pan/zoom
function getVisibleRange(length) {
    // x = (i * width / length) * zoomLevel - panX is on-screen when 0 <= x < width
    // Solve for i: 0 <= (i * width / length) * zoomLevel - panX < width
    // => panX <= (i * width / length) * zoomLevel < panX + width
    // => (panX) * (length / (width * zoomLevel)) <= i < (panX + width) * (length / (width * zoomLevel))
    const denom = (width * zoomLevel);
    const factor = denom !== 0 ? (length / denom) : length;
    let start = Math.floor(Math.max(0, panX * factor));
    let end = Math.ceil(Math.min(length, (panX + width) * factor));
    if (!isFinite(start) || !isFinite(end)) { start = 0; end = length; }
    if (start < 0) start = 0;
    if (end > length) end = length;
    if (start >= end) { start = 0; end = Math.min(length, start + 1); }
    return { start, end };
}

function draw() {
    background(255);

    if (!analysisData || !csvData) {
        drawNoDataMessage();
        return;
    }

    // Draw grid
    drawGrid();

    // Draw original signal if enabled
    if (showOriginal && signalBuffer.length > 0) {
        drawSignal();
    }

    // Draw reconstruction and residual if enabled
    if (showReconstruction && reconstructionBuffer.length > 0) {
        drawReconstruction();
    }
    if (showResidual && residualBuffer.length > 0) {
        drawResidual();
    }

    // Draw chirplets
    if (showChirplets) {
        drawChirplets();
    }
    if (showFInst && showChirplets) {
        drawInstantaneousFrequency();
    }

    // Draw info
    drawInfo();

    // Draw playback marker if audio is playing
    drawPlaybackMarker();
}

function drawNoDataMessage() {
    fill(0);
    textAlign(CENTER, CENTER);
    textSize(24);
    text('Load a JSON analysis file to visualize', width/2, height/2);
}

function drawGrid() {
    // Light gray grid in HSB
    stroke(0, 0, 85);
    strokeWeight(1);

    // Vertical grid lines
    for (let x = 0; x < width; x += 50 * zoomLevel) {
        line(x - panX, 0, x - panX, height);
    }

    // Horizontal grid lines
    for (let y = height/2; y < height; y += 50) {
        line(0, y, width, y);
    }
    for (let y = height/2; y > 0; y -= 50) {
        line(0, y, width, y);
    }
}

function drawSignal() {
    // Blue tone in HSB
    stroke(220, 80, 90);
    strokeWeight(2);
    noFill();

    const { start, end } = getVisibleRange(signalBuffer.length);
    const count = Math.max(1, end - start);
    const maxVerts = Math.floor(width * 1.5); // decimate to ~1.5 vertices per screen pixel
    const step = Math.max(1, Math.floor(count / maxVerts));

    beginShape();
    for (let i = start; i < end; i += step) {
        let x = (i * width / signalBuffer.length) * zoomLevel - panX;
        let y = height/2 - signalBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    // ensure last point in range is drawn
    if ((end - 1) > start) {
        const i = end - 1;
        let x = (i * width / signalBuffer.length) * zoomLevel - panX;
        let y = height/2 - signalBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    endShape();
}

function drawReconstruction() {
    stroke(120, 80, 60); // green-ish in HSB
    strokeWeight(2);
    noFill();

    const { start, end } = getVisibleRange(reconstructionBuffer.length);
    const count = Math.max(1, end - start);
    const maxVerts = Math.floor(width * 1.5);
    const step = Math.max(1, Math.floor(count / maxVerts));

    beginShape();
    for (let i = start; i < end; i += step) {
        let x = (i * width / reconstructionBuffer.length) * zoomLevel - panX;
        let y = height/2 - reconstructionBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    if ((end - 1) > start) {
        const i = end - 1;
        let x = (i * width / reconstructionBuffer.length) * zoomLevel - panX;
        let y = height/2 - reconstructionBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    endShape();
}

function drawResidual() {
    stroke(0, 0, 40); // gray
    strokeWeight(1);
    noFill();

    const { start, end } = getVisibleRange(residualBuffer.length);
    const count = Math.max(1, end - start);
    const maxVerts = Math.floor(width * 1.5);
    const step = Math.max(1, Math.floor(count / maxVerts));

    beginShape();
    for (let i = start; i < end; i += step) {
        let x = (i * width / residualBuffer.length) * zoomLevel - panX;
        let y = height/2 - residualBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    if ((end - 1) > start) {
        const i = end - 1;
        let x = (i * width / residualBuffer.length) * zoomLevel - panX;
        let y = height/2 - residualBuffer[i] * scaleFactorSignal;
        vertex(x, y);
    }
    endShape();
}

function drawChirplets() {
    for (let i = 0; i < chirpletBuffers.length; i++) {
        if (!chirpletVisibility[i]) continue;

        let chirplet = analysisData.result.chirplets[i];
        let colorHue = (i * 137.5) % 360; // Golden angle for distinct colors

        stroke(colorHue, 80, 80);
        strokeWeight(1.5);
        noFill();

        let buffer = chirpletBuffers[i];
        const { start, end } = getVisibleRange(buffer.length);
        const count = Math.max(1, end - start);
        const maxVerts = Math.floor(width * 1.2); // slightly fewer vertices for chirplets
        const step = Math.max(1, Math.floor(count / maxVerts));

        beginShape();
        for (let j = start; j < end; j += step) {
            let x = (j * width / buffer.length) * zoomLevel - panX;
            let y = height/2 - buffer[j] * scaleFactorSignal; // Use same scale factor as original signal
            vertex(x, y);
        }
        if ((end - 1) > start) {
            const j = end - 1;
            let x = (j * width / buffer.length) * zoomLevel - panX;
            let y = height/2 - buffer[j] * scaleFactorSignal;
            vertex(x, y);
        }
        endShape();

        // Draw chirplet center marker (handle combined multi-window JSON)
        let tc_samples = chirplet.time_center_samples;
        if (analysisData.analysis_type === 'multi_sample_combined') {
            tc_samples = tc_samples - analysisData.start_sample;
        }
        let centerX = (tc_samples / signalBuffer.length) * width * zoomLevel - panX;
        let centerY = height/2;
        fill(colorHue, 80, 80);
        noStroke();
        ellipse(centerX, centerY, 8, 8);
    }
}

function drawInfo() {
    fill(0);
    textAlign(LEFT, TOP);
    textSize(12);

    let info = [];
    if (analysisData && csvData) {
        info.push(`File: ${analysisData.source_file}`);
        info.push(`Column: ${analysisData.column_name} (${analysisData.column_index})`);
        info.push(`Samples: ${analysisData.num_samples}`);
        info.push(`Sampling Rate: ${analysisData.sampling_frequency} Hz`);
        info.push(`Analysis Type: ${analysisData.analysis_type}`);
        info.push(`Chirplets: ${analysisData.result.chirplets.length}`);
        info.push(`Zoom: ${zoomLevel.toFixed(1)}x`);
        info.push(`Pan X: ${panX}`);
        if (signalEnergy > 0) {
            const explained = reconstructionEnergy / signalEnergy * 100;
            info.push(`Signal Energy: ${signalEnergy.toFixed(4)}`);
            info.push(`Reconstruction Energy: ${reconstructionEnergy.toFixed(4)} (${explained.toFixed(2)}%)`);
            info.push(`Residual Energy: ${residualEnergy.toFixed(4)} (${(100 - explained).toFixed(2)}%)`);
        }
    }

    for (let i = 0; i < info.length; i++) {
        text(info[i], 10, 10 + i * 15);
    }
}

function drawPlaybackMarker() {
    // Requires chirpletAudio module
    if (!window.chirpletAudio || typeof chirpletAudio.getIsPlaying !== 'function') return;
    if (!chirpletAudio.getIsPlaying()) return;
    if (!analysisData || !csvData || signalBuffer.length === 0) return;

    const fs = analysisData.sampling_frequency || 1;
    const posSec = chirpletAudio.getPlaybackPositionSec(); // relative to segment start
    const posSamples = Math.max(0, Math.min(signalBuffer.length - 1, Math.round(posSec * fs)));

    // Map to canvas x similar to other series
    const x = (posSamples * width / signalBuffer.length) * zoomLevel - panX;

    // Draw a red vertical line marker
    push();
    stroke(0, 80, 80); // red in HSB
    strokeWeight(2);
    line(x, 0, x, height);

    // Draw small triangle at the top as a playhead indicator
    noStroke();
    fill(0, 80, 80);
    const triSize = 8;
    triangle(x - triSize, 0, x + triSize, 0, x, triSize * 1.8);

    // Time label
    const label = `${posSec.toFixed(3)} s`;
    textAlign(CENTER, TOP);
    textSize(11);
    fill(0);
    // Draw label slightly below the triangle without going off canvas
    const tx = Math.max(25, Math.min(width - 25, x));
    text(label, tx, triSize * 2 + 2);
    pop();
}

function loadJSONFile() {
    const fileInput = document.getElementById('jsonFile');
    if (fileInput.files.length === 0) {
        alert('Please select a JSON file first');
        return;
    }

    const file = fileInput.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        try {
            analysisData = JSON.parse(e.target.result);
            // Load the corresponding CSV file
            loadCSVFile(analysisData.source_file);
        } catch (error) {
            alert('Error parsing JSON file: ' + error.message);
        }
    };

    reader.readAsText(file);
}

async function loadCSVFile(csvPath) {
    // First attempt: fetch via HTTP using relative path from project root
    // Since index.html is served from /p5js/, prefix with '/' so 'data/...' resolves to '/data/...'
    const url = csvPath.startsWith('/') ? csvPath : ('/' + csvPath);
    try {
        const resp = await fetch(url, { cache: 'no-cache' });
        if (resp.ok) {
            const text = await resp.text();
            processCSVData(text);
            return; // success
        }
        console.warn('Fetch CSV failed with status', resp.status, '— falling back to manual file selection.');
    } catch (e) {
        console.warn('Fetch CSV threw error — falling back to manual file selection.', e);
    }

    // Fallback: prompt user to select the CSV file manually
    const csvInput = document.createElement('input');
    csvInput.type = 'file';
    csvInput.accept = '.csv';
    csvInput.style.display = 'none';
    csvInput.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            const file = e.target.files[0];
            const reader = new FileReader();
            reader.onload = function(e) {
                processCSVData(e.target.result);
            };
            reader.readAsText(file);
        }
    });

    // Auto-trigger file selection or provide instructions
    alert(`Could not fetch CSV automatically. Please select the CSV file: ${csvPath}`);
    csvInput.click();
}

function processCSVData(csvText) {
    // Parse CSV data
    const lines = csvText.trim().split('\n');
    if (lines.length === 0) {
        alert('CSV file is empty');
        return;
    }

    // Parse header
    const headers = lines[0].split(',').map(h => h.trim());

    // Check if column index is valid
    if (analysisData.column_index >= headers.length) {
        alert(`Invalid column index ${analysisData.column_index}. CSV has ${headers.length} columns.`);
        return;
    }

    // Extract data from specified column
    const rawData = [];
    for (let i = 1; i < lines.length; i++) {
        const cells = lines[i].split(',');
        if (cells.length > analysisData.column_index) {
            const value = parseFloat(cells[analysisData.column_index].trim());
            if (!isNaN(value)) {
                rawData.push(value);
            }
        }
    }

    // Extract the specified sample range
    const startIdx = Math.max(0, analysisData.start_sample);
    const endIdx = Math.min(rawData.length, startIdx + analysisData.num_samples);
    const selectedData = rawData.slice(startIdx, endIdx);

    if (selectedData.length === 0) {
        alert('No valid data found in the specified range');
        return;
    }

    csvData = selectedData;

    // Apply DC offset removal (same as C++ code)
    const sum = selectedData.reduce((a, b) => a + b, 0);
    const mean = sum / selectedData.length;
    signalBuffer = selectedData.map(val => val - mean);
    signalEnergy = signalBuffer.reduce((acc, v) => acc + v * v, 0);

    // Compute and cache y-axis scale factor once (fit ~80% of canvas height)
    let maxAmplitude = 0;
    for (let v of signalBuffer) {
        const a = Math.abs(v);
        if (a > maxAmplitude) maxAmplitude = a;
    }
    scaleFactorSignal = maxAmplitude > 0 ? (height * 0.4) / maxAmplitude : 1;

    // Process analysis data
    processAnalysisData();

    // Update UI
    updateChirpletToggles();
    updateAnalysisInfo();
}

function processAnalysisData() {
    // Generate chirplet signals
    chirpletBuffers = [];
    chirpletVisibility = new Array(analysisData.result.chirplets.length).fill(true);

    // Attach window metadata for combined analysis
    const isCombined = analysisData.analysis_type === 'multi_sample_combined';
    const chirpsPerWindow = analysisData.num_chirps_per_window || 0;
    const windowStarts = analysisData.window_starts || [];
    const windowSize = analysisData.window_size || 0;

    for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        const chirplet = analysisData.result.chirplets[i];
        if (isCombined && chirpsPerWindow > 0 && windowStarts.length > 0 && windowSize > 0) {
            const wIdx = Math.floor(i / chirpsPerWindow);
            chirplet.__windowIndex = wIdx;
            chirplet.__windowStart = windowStarts[wIdx]; // absolute sample index
            chirplet.__windowSize = windowSize;
        }
        let buffer = generateChirpletSignal(chirplet);
        chirpletBuffers.push(buffer);
    }

    // Build reconstruction buffer as sum of chirplets
    reconstructionBuffer = new Array(signalBuffer.length).fill(0);
    for (let buf of chirpletBuffers) {
        for (let i = 0; i < reconstructionBuffer.length; i++) {
            reconstructionBuffer[i] += buf[i];
        }
    }

    // If combined multi-window analysis, average contributions in overlapping regions by window coverage
    if (isCombined && windowStarts.length > 0 && windowSize > 0) {
        const weights = new Array(signalBuffer.length).fill(0);
        for (let w = 0; w < windowStarts.length; w++) {
            const localStart = Math.max(0, (windowStarts[w] - analysisData.start_sample) | 0);
            const localEnd = Math.min(signalBuffer.length, localStart + windowSize);
            for (let i = localStart; i < localEnd; i++) weights[i] += 1;
        }
        for (let i = 0; i < reconstructionBuffer.length; i++) {
            if (weights[i] > 0) reconstructionBuffer[i] /= weights[i];
        }
    }

    // Build residual buffer: original - reconstruction
    residualBuffer = new Array(signalBuffer.length);
    for (let i = 0; i < signalBuffer.length; i++) {
        residualBuffer[i] = signalBuffer[i] - reconstructionBuffer[i];
    }

    // Energies
    reconstructionEnergy = reconstructionBuffer.reduce((acc, v) => acc + v * v, 0);
    residualEnergy = residualBuffer.reduce((acc, v) => acc + v * v, 0);
}

function generateChirpletSignal(chirplet) {
    // Generate a Gaussian-enveloped chirplet signal matching ACT::g()
    let fs = analysisData.sampling_frequency;
    let length = signalBuffer.length;

    // Convert global to local sample index for combined JSONs
    let tc_samples = chirplet.time_center_samples;
    if (analysisData.analysis_type === 'multi_sample_combined') {
        tc_samples = tc_samples - analysisData.start_sample;
    }
    let fc = chirplet.frequency_hz;
    let c = chirplet.chirp_rate_hz_per_s;
    let coeff = chirplet.coefficient;

    // In C++: Dt = exp(logDt). JSON provides duration_ms = 1000 * exp(logDt)
    let Dt = chirplet.duration_ms / 1000.0; // seconds

    // Build unnormalized atom and compute normalization energy
    let atom = new Array(length);
    let energy = 0;
    // If this is a combined analysis, compute normalization energy over the chirplet's own window only
    let normStart = 0;
    let normEnd = length; // exclusive
    if (analysisData.analysis_type === 'multi_sample_combined' && chirplet.__windowSize && (chirplet.__windowStart !== undefined)) {
        // Convert absolute window start to local index
        normStart = Math.max(0, (chirplet.__windowStart - analysisData.start_sample) | 0);
        normEnd = Math.min(length, normStart + chirplet.__windowSize);
    }

    for (let n = 0; n < length; n++) {
        let t = n / fs;
        let t_center = tc_samples / fs;
        let t_rel = t - t_center;

        // Gaussian envelope: exp(-0.5 * (t_rel / Dt)^2)
        let envelope = Math.exp(-0.5 * Math.pow(t_rel / Dt, 2));

        // Phase centered at tc: 2π [c * t_rel^2 + fc * t_rel]
        let phase = 2 * Math.PI * (c * t_rel * t_rel + fc * t_rel);
        let value = envelope * Math.cos(phase);
        if (!isFinite(value)) value = 0;
        atom[n] = value;
        if (n >= normStart && n < normEnd) energy += value * value;
    }

    // L2-normalize atom to unit energy over the appropriate window, then scale by coefficient
    let invNorm = energy > 0 ? 1 / Math.sqrt(energy) : 0;
    let buffer = new Array(length);
    for (let n = 0; n < length; n++) {
        buffer[n] = atom[n] * invNorm * coeff;
    }

    // For combined multi-window files, zero out contributions outside the chirplet's own window
    if (analysisData.analysis_type === 'multi_sample_combined' && chirplet.__windowSize && (chirplet.__windowStart !== undefined)) {
        const localStart = Math.max(0, (chirplet.__windowStart - analysisData.start_sample) | 0);
        const localEnd = Math.min(length, localStart + chirplet.__windowSize);
        for (let n = 0; n < localStart; n++) buffer[n] = 0;
        for (let n = localEnd; n < length; n++) buffer[n] = 0;
    }

    return buffer;
}

function updateChirpletToggles() {
    const container = document.getElementById('chirpletToggles');
    container.innerHTML = '';

    for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        const chirplet = analysisData.result.chirplets[i];
        const div = document.createElement('div');
        div.className = 'chirplet-toggle';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `chirplet-${i}`;
        checkbox.checked = chirpletVisibility[i];
        checkbox.addEventListener('change', () => {
            chirpletVisibility[i] = checkbox.checked;
        });

        const label = document.createElement('label');
        label.htmlFor = `chirplet-${i}`;
        const amp = chirplet.coefficient;
        const energyPct = signalEnergy > 0 ? (100 * (amp * amp) / signalEnergy) : 0;
        // Use global list index to avoid confusion from repeated per-window .index values
        let prefix = `Chirplet #${i+1}`;
        if (analysisData.analysis_type === 'multi_sample_combined' && typeof chirplet.__windowIndex === 'number') {
            prefix += ` (W${chirplet.__windowIndex+1})`;
        }
        label.textContent = `${prefix}: ${chirplet.frequency_hz.toFixed(3)} Hz @ ${chirplet.time_center_seconds.toFixed(6)} s | coeff=${amp.toFixed(6)} | energy=${energyPct.toFixed(2)}%`;

        div.appendChild(checkbox);
        div.appendChild(label);
        container.appendChild(div);
    }
}

function updateAnalysisInfo() {
    const infoDiv = document.getElementById('analysisInfo');
    if (analysisData) {
        let html = `
            <strong>Source:</strong> ${analysisData.source_file}<br>
            <strong>Column:</strong> ${analysisData.column_name} (${analysisData.column_index})<br>
            <strong>Samples:</strong> ${analysisData.num_samples}<br>
            <strong>Sampling Rate:</strong> ${analysisData.sampling_frequency} Hz<br>
            <strong>Analysis Type:</strong> ${analysisData.analysis_type}<br>
            <strong>Chirplets Found:</strong> ${analysisData.result.chirplets.length}<br>
        `;
        if (analysisData.result && typeof analysisData.result.error === 'number') {
            html += `<strong>Final Error:</strong> ${analysisData.result.error.toFixed(6)}<br>`;
        } else if (analysisData.result && Array.isArray(analysisData.result.error_per_window)) {
            const errs = analysisData.result.error_per_window;
            if (errs.length > 0) {
                const avg = errs.reduce((a, b) => a + b, 0) / errs.length;
                html += `<strong>Avg Window Error:</strong> ${avg.toFixed(6)}<br>`;
            }
        }
        infoDiv.innerHTML = html;
    } else {
        infoDiv.textContent = 'No data loaded';
    }
}

function updateZoom() {
    zoomLevel = parseFloat(document.getElementById('zoom').value);
    document.getElementById('zoomValue').textContent = zoomLevel.toFixed(1) + 'x';
}

function updatePan() {
    panX = parseInt(document.getElementById('panX').value);
    document.getElementById('panXValue').textContent = panX;
}

function updateShowOriginal() {
    showOriginal = document.getElementById('showOriginal').checked;
}

function updateShowReconstruction() {
    showReconstruction = document.getElementById('showReconstruction').checked;
}

function updateShowResidual() {
    showResidual = document.getElementById('showResidual').checked;
}

function updateShowFInst() {
    showFInst = document.getElementById('showFInst').checked;
}

function drawInstantaneousFrequency() {
    // Draw f_inst(t) = fc + 2 c (t - tc) as a thin ridge for each visible chirplet
    // We map frequency to a vertical offset above the midline using a compact scale
    const fs = analysisData.sampling_frequency;
    const length = signalBuffer.length;

    // Frequency-to-pixels scale for the overlay (compact): 1 Hz => 6 px
    const hzScale = 6;

    for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        if (!chirpletVisibility[i]) continue;
        const chir = analysisData.result.chirplets[i];

        // Window bounds for combined files
        let localStart = 0;
        let localEnd = length;
        if (analysisData.analysis_type === 'multi_sample_combined' && chir.__windowSize && (chir.__windowStart !== undefined)) {
            localStart = Math.max(0, (chir.__windowStart - analysisData.start_sample) | 0);
            localEnd = Math.min(length, localStart + chir.__windowSize);
        }

        // Compute tc in samples (local)
        let tc_samples = chir.time_center_samples;
        if (analysisData.analysis_type === 'multi_sample_combined') tc_samples -= analysisData.start_sample;

        const fc = chir.frequency_hz;
        const c = chir.chirp_rate_hz_per_s; // Hz/s

        // Visual style
        const colorHue = (i * 137.5) % 360;
        stroke(colorHue, 80, 50, 0.8); // semi-transparent
        strokeWeight(1);
        noFill();

        beginShape();
        for (let n = localStart; n < localEnd; n++) {
            const t = n / fs;
            const t_center = tc_samples / fs;
            const f_inst = fc + 2 * c * (t - t_center); // Hz (can be negative)
            // Map time to x
            const x = (n * width / length) * zoomLevel - panX;
            // Map frequency magnitude to a small vertical offset above midline
            const y = height/2 - Math.sign(f_inst) * Math.min(Math.abs(f_inst) * hzScale, height * 0.45);
            vertex(x, y);
        }
        endShape();
    }
}

function updateShowChirplets() {
    const el = document.getElementById('showChirplets');
    showChirplets = el ? el.checked : true;
}

function showAllChirplets() {
    if (!analysisData || !analysisData.result) return;
    chirpletVisibility = chirpletVisibility.map(() => true);
    // Update checkboxes in the DOM
    for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        const cb = document.getElementById(`chirplet-${i}`);
        if (cb) cb.checked = true;
    }
}

function hideAllChirplets() {
    if (!analysisData || !analysisData.result) return;
    chirpletVisibility = chirpletVisibility.map(() => false);
    // Update checkboxes in the DOM
    for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        const cb = document.getElementById(`chirplet-${i}`);
        if (cb) cb.checked = false;
    }
}

// Mouse wheel for zoom - DISABLED, use sliders only
// function mouseWheel(event) {
//     if (event.delta > 0) {
//         zoomLevel = Math.max(0.1, zoomLevel - 0.1);
//     } else {
//         zoomLevel = Math.min(10, zoomLevel + 0.1);
//     }
//     document.getElementById('zoom').value = zoomLevel;
//     document.getElementById('zoomValue').textContent = zoomLevel.toFixed(1) + 'x';
//     return false;
// }

// Mouse drag for pan
let isDragging = false;
let lastMouseX = 0;

function mousePressed() {
    // Only start dragging if the press started over the canvas and left button
    if (overCanvas && (mouseButton === LEFT)) {
        isDragging = true;
        lastMouseX = mouseX;
    }
}

function mouseDragged() {
    if (isDragging) {
        let deltaX = mouseX - lastMouseX;
        panX -= deltaX / zoomLevel;
        const panSlider = document.getElementById('panX');
        const panLabel = document.getElementById('panXValue');
        if (panSlider) panSlider.value = panX;
        if (panLabel) panLabel.textContent = panX;
        lastMouseX = mouseX;
    }
}

function mouseReleased() {
    isDragging = false;
}
