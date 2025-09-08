// Chirplet audio synthesizer module (separate from sketch.js)
// Uses Web Audio (via p5.getAudioContext()) and generates time-domain chirplet samples
// to preserve signed instantaneous frequency sweeps.

(function(global){
  const Audio = {
    ctx: null,
    masterGain: null,
    compressor: null,
    limiterEnabled: false,
    pitchScale: 100.0,
    coverageK: 3.0,
    mixGain: 3.0, // additional mix boost factor
    exportFrequencyMode: 'edge_zero', // 'signed' | 'shift_up' | 'edge_zero' | 'mirror' (export only)
    playVisibleOnly: true,
    sources: [],
    isPlaying: false,
    baseStartTime: 0,
    masterVolumeVal: 0.25,
  };

  function getCtx() {
    if (Audio.ctx) return Audio.ctx;
    if (typeof getAudioContext === 'function') {
      Audio.ctx = getAudioContext();
    } else {
      const AC = window.AudioContext || window.webkitAudioContext;
      Audio.ctx = new AC();
    }
    return Audio.ctx;
  }
  function audioBufferToWav(buffer) {
    // 16-bit PCM WAV encoder
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const samples = buffer.length;
    const bytesPerSample = 2;
    const blockAlign = numChannels * bytesPerSample;
    const dataSize = samples * blockAlign;
    const bufferSize = 44 + dataSize;
    const ab = new ArrayBuffer(bufferSize);
    const view = new DataView(ab);

    function writeString(offset, str) {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    }
    function writePCM16(output, offset, input) {
      for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
    }

    // RIFF header
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataSize, true);
    writeString(8, 'WAVE');
    // fmt chunk
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // PCM chunk size
    view.setUint16(20, 1, true);  // Audio format = PCM
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    // data chunk
    writeString(36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave if stereo; our renderer is mono (1 channel), but keep generic
    const channels = [];
    for (let ch = 0; ch < numChannels; ch++) channels.push(buffer.getChannelData(ch));

    let offset = 44;
    if (numChannels === 1) {
      writePCM16(view, offset, channels[0]);
    } else {
      // interleave
      for (let i = 0; i < samples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
          let s = Math.max(-1, Math.min(1, channels[ch][i]));
          view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
          offset += 2;
        }
      }
    }

    return new Blob([view], { type: 'audio/wav' });
  }

  async function exportWav(analysisData, chirpletVisibility) {
    console.log('[Export] Starting exportWav...');
    if (!analysisData || !analysisData.result || !analysisData.result.chirplets) {
      console.error('[Export] No analysis data or chirplets. Aborting export.');
      return;
    }
    const realtimeCtx = getCtx();
    const sampleRate = realtimeCtx.sampleRate || 44100;
    console.log(`[Export] Realtime sampleRate=${sampleRate}, masterVolume=${Audio.masterVolumeVal}, limiter=${Audio.limiterEnabled}, pitchScale=${Audio.pitchScale}, coverageK=${Audio.coverageK}, mixGain=${Audio.mixGain}, freqMode=${Audio.exportFrequencyMode}`);

    // Reuse scheduling logic to compute timebase and selection
    const fs = analysisData.sampling_frequency;
    const isCombined = analysisData.analysis_type === 'multi_sample_combined';
    const localDur = analysisData.num_samples / fs;
    let maxTc = 0;
    for (const ch of (analysisData.result.chirplets || [])) {
      const tcs = Number(ch.time_center_seconds) || 0;
      if (tcs > maxTc) maxTc = tcs;
    }
    const timesAreGlobal = isCombined || (maxTc > localDur * 1.25);
    const segmentStartSec = timesAreGlobal ? (analysisData.start_sample / fs) : 0;
    const segmentEndSec = segmentStartSec + (analysisData.num_samples / fs);
    console.log(`[Export] Timebase: isCombined=${isCombined}, timesAreGlobal=${timesAreGlobal}, segment=${segmentStartSec}..${segmentEndSec}`);
    const k = Audio.coverageK;

    const all = analysisData.result.chirplets;
    const selected = [];
    for (let i = 0; i < all.length; i++) {
      if (Audio.playVisibleOnly && !chirpletVisibility[i]) continue;
      selected.push({ ch: all[i], i });
    }
    if (selected.length === 0) {
      console.warn('[Export] No chirplets selected (Play Visible Only may exclude all). Aborting export.');
      return;
    }
    console.log(`[Export] Selected ${selected.length} chirplets for rendering.`);

    // Normalization
    let sumCoeff2 = 0;
    for (const { ch } of selected) sumCoeff2 += (ch.coefficient * ch.coefficient);
    const coeffNorm = Math.sqrt(Math.max(1e-12, sumCoeff2));

    // Determine total offline duration
    const base = 0.2;
    let maxEndRel = 0;
    const clampedWindows = [];
    for (const { ch, i } of selected) {
      const Dt = ch.duration_ms / 1000.0;
      const tc = ch.time_center_seconds;
      let t0 = tc - k * Dt;
      let t1 = tc + k * Dt;
      if (!timesAreGlobal) {
        t0 = Math.max(t0, segmentStartSec);
        t1 = Math.min(t1, segmentEndSec);
      }
      if (isCombined && typeof ch.__windowStart === 'number' && typeof ch.__windowSize === 'number') {
        const ws = ch.__windowStart / fs;
        const we = ws + ch.__windowSize / fs;
        t0 = Math.max(t0, ws);
        t1 = Math.min(t1, we);
      }
      if (t1 <= t0) {
        console.warn(`[Export] Skipping chirplet ${i} due to empty window after clamping.`);
        continue;
      }
      const startOffset = Math.max(0, (t0 - segmentStartSec));
      const endRel = startOffset + (t1 - t0);
      if (endRel > maxEndRel) maxEndRel = endRel;
      clampedWindows.push({ ch, t0, t1, startOffset });
    }
    const totalDuration = base + maxEndRel + 0.1;
    console.log(`[Export] totalDuration=${totalDuration.toFixed(3)}s, voices to render=${clampedWindows.length}`);

    // Create OfflineAudioContext and schedule
    const offline = new OfflineAudioContext(1, Math.ceil(totalDuration * sampleRate), sampleRate);
    console.log(`[Export] OfflineAudioContext created: length=${Math.ceil(totalDuration * sampleRate)}, sampleRate=${sampleRate}`);
    const master = offline.createGain();
    master.gain.setValueAtTime(Audio.masterVolumeVal, 0);
    let destNode = master;
    if (Audio.limiterEnabled) {
      const comp = offline.createDynamicsCompressor();
      // Configure limiter on offline context
      if (comp.threshold) comp.threshold.setValueAtTime(-6, 0);
      if (comp.knee) comp.knee.setValueAtTime(12, 0);
      if (comp.ratio) comp.ratio.setValueAtTime(6, 0);
      if (comp.attack) comp.attack.setValueAtTime(0.003, 0);
      if (comp.release) comp.release.setValueAtTime(0.1, 0);
      master.connect(comp);
      destNode = comp;
    }
    destNode.connect(offline.destination);

    // Schedule voices
    for (const win of clampedWindows) {
      const { ch, t0, t1, startOffset } = win;
      const buffer = makeChirpletBuffer(ch, t0, t1, fs, Audio.pitchScale, offline, Audio.exportFrequencyMode);
      const source = offline.createBufferSource();
      source.buffer = buffer;

      const voiceGain = offline.createGain();
      const amp = (Math.abs(ch.coefficient) / coeffNorm) * Audio.mixGain;
      voiceGain.gain.setValueAtTime(amp, 0);

      source.connect(voiceGain);
      voiceGain.connect(master);

      const when = base + startOffset;
      source.start(when);
      source.stop(when + buffer.duration);
    }
    console.log('[Export] Scheduling complete. Starting offline rendering...');

    let rendered;
    try {
      rendered = await offline.startRendering();
      console.log('[Export] Offline rendering finished. Duration:', rendered.duration, 'seconds');
    } catch (e) {
      console.error('[Export] Offline rendering failed:', e);
      throw e;
    }
    const blob = audioBufferToWav(rendered);
    console.log('[Export] WAV blob created. Size (bytes):', blob.size);

    // Download
    const a = document.createElement('a');
    const url = URL.createObjectURL(blob);
    a.href = url;
    const baseName = (analysisData.source_file || 'act_synth').split(/[\\/]/).pop().replace(/\.[^/.]+$/, '');
    a.download = `${baseName}_chirplet_synth.wav`;
    document.body.appendChild(a);
    try {
      a.click();
      console.log('[Export] Download triggered via anchor click.');
    } catch (e) {
      console.warn('[Export] Anchor click failed, opening URL in new tab as fallback.');
      window.open(url, '_blank');
    }
    setTimeout(() => {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 0);
  }

  function configureLimiter(node, opts={}) {
    // Defaults are gentle and act like a soft limiter
    const threshold = opts.threshold ?? -6; // dB
    const knee = opts.knee ?? 12;           // dB
    const ratio = opts.ratio ?? 6;          // :1
    const attack = opts.attack ?? 0.003;    // seconds
    const release = opts.release ?? 0.1;    // seconds
    if (node.threshold) node.threshold.setValueAtTime(threshold, Audio.ctx.currentTime);
    if (node.knee) node.knee.setValueAtTime(knee, Audio.ctx.currentTime);
    if (node.ratio) node.ratio.setValueAtTime(ratio, Audio.ctx.currentTime);
    if (node.attack) node.attack.setValueAtTime(attack, Audio.ctx.currentTime);
    if (node.release) node.release.setValueAtTime(release, Audio.ctx.currentTime);
  }

  function connectChain() {
    const ctx = getCtx();
    try { Audio.masterGain.disconnect(); } catch(e) {}
    try { Audio.compressor && Audio.compressor.disconnect(); } catch(e) {}
    if (Audio.limiterEnabled) {
      Audio.masterGain.connect(Audio.compressor);
      Audio.compressor.connect(ctx.destination);
    } else {
      Audio.masterGain.connect(ctx.destination);
    }
  }

  function initAudioEngine() {
    const ctx = getCtx();
    if (!Audio.masterGain) {
      Audio.masterGain = ctx.createGain();
      Audio.masterGain.gain.value = 0.25; // -12 dB headroom
    }
    if (!Audio.compressor) {
      Audio.compressor = ctx.createDynamicsCompressor();
      configureLimiter(Audio.compressor, {});
    }
    connectChain();
  }

  function setMasterVolume(linear) {
    initAudioEngine();
    const v = Math.max(0, Math.min(1, Number(linear) || 0));
    const t = getCtx().currentTime;
    Audio.masterGain.gain.setTargetAtTime(v, t, 0.01);
    Audio.masterVolumeVal = v;
  }

  function setLimiterEnabled(enabled) {
    initAudioEngine();
    Audio.limiterEnabled = !!enabled;
    connectChain();
  }

  function setPitchScale(scale) {
    Audio.pitchScale = Math.max(1, Number(scale) || 1);
  }

  function setCoverageK(k) {
    Audio.coverageK = Math.max(0.5, Number(k) || 3.0);
  }

  function setPlayVisibleOnly(b) {
    Audio.playVisibleOnly = !!b;
  }

  function setMixGain(g) {
    const val = Math.max(0, Number(g) || 1);
    Audio.mixGain = val;
  }

  function setExportFrequencyMode(mode) {
    const allowed = ['signed', 'shift_up', 'edge_zero', 'mirror'];
    if (!allowed.includes(mode)) {
      console.warn('[Export] Unknown frequency mode, defaulting to edge_zero:', mode);
      Audio.exportFrequencyMode = 'edge_zero';
    } else {
      Audio.exportFrequencyMode = mode;
    }
    console.log('[Export] Set frequency mode to:', Audio.exportFrequencyMode);
  }

  function stopAll() {
    const ctx = getCtx();
    const now = ctx.currentTime;
    for (const v of Audio.sources) {
      try { v.source.stop(now); } catch(e) {}
      try { v.source.disconnect(); } catch(e) {}
      try { v.gain && v.gain.disconnect(); } catch(e) {}
    }
    Audio.sources = [];
    Audio.isPlaying = false;
  }

  function makeChirpletBuffer(ch, tStart, tEnd, fsSignal, pitchScale, ctxOverride, freqMode = 'signed') {
    const ctx = ctxOverride || getCtx();
    const fsAudio = ctx.sampleRate;
    const duration = Math.max(0, tEnd - tStart);
    const N = Math.max(1, Math.round(duration * fsAudio));
    const buffer = ctx.createBuffer(1, N, fsAudio);
    const data = buffer.getChannelData(0);

    // Parameters
    const tc_sec = ch.time_center_seconds;
    const Dt = ch.duration_ms / 1000.0;
    const fc0 = ch.frequency_hz * pitchScale;
    const c0 = ch.chirp_rate_hz_per_s * pitchScale; // scales f_inst by pitchScale

    // Prepare frequency behavior per mode
    const tRelStart = tStart - tc_sec;
    const tRelEnd = tEnd - tc_sec;
    let S = 0; // constant shift in Hz for shift-based modes
    let mirror = false;
    switch (freqMode) {
      case 'signed':
        // no change
        break;
      case 'shift_up': {
        const fAtStart = fc0 + 2 * c0 * tRelStart;
        const fAtEnd = fc0 + 2 * c0 * tRelEnd;
        const fMin = (c0 >= 0) ? fAtStart : fAtEnd; // linear, min at one edge
        S = Math.max(0, -fMin);
        break;
      }
      case 'edge_zero': {
        if (c0 >= 0) {
          const fAtStart = fc0 + 2 * c0 * tRelStart;
          S = -fAtStart; // starts at 0 Hz
        } else {
          const fAtEnd = fc0 + 2 * c0 * tRelEnd;
          S = -fAtEnd; // ends at 0 Hz
        }
        break;
      }
      case 'mirror':
        mirror = true;
        break;
      default:
        break;
    }

    // Synthesis
    const twoPi = 2 * Math.PI;
    let peak = 0.0;
    if (!mirror) {
      // Closed-form phase with optional constant shift S
      for (let n = 0; n < N; n++) {
        const t = tStart + n / fsAudio;
        const t_rel = t - tc_sec;
        const env = Math.exp(-0.5 * Math.pow(t_rel / Dt, 2));
        const phase = twoPi * (c0 * t_rel * t_rel + (fc0 + S) * t_rel);
        const val = env * Math.cos(phase);
        data[n] = val;
        const a = Math.abs(val);
        if (a > peak) peak = a;
      }
    } else {
      // Mirror mode: integrate |f_inst(t)| numerically for phase continuity
      let phase = 0;
      let prevInst = Math.abs(fc0 + 2 * c0 * (tRelStart));
      for (let n = 0; n < N; n++) {
        const t = tStart + n / fsAudio;
        const t_rel = t - tc_sec;
        const env = Math.exp(-0.5 * Math.pow(t_rel / Dt, 2));
        const inst = Math.abs(fc0 + 2 * c0 * t_rel);
        // Trapezoidal increment keeps phase smooth near the zero
        const avg = (prevInst + inst) * 0.5;
        phase += twoPi * (avg / fsAudio);
        const val = env * Math.cos(phase);
        data[n] = val;
        const a = Math.abs(val);
        if (a > peak) peak = a;
        prevInst = inst;
      }
    }

    // Peak normalization with 0.9 headroom
    if (peak > 0) {
      const inv = 0.9 / peak;
      for (let n = 0; n < N; n++) data[n] *= inv;
    }

    // Short fades to avoid clicks
    const fadeIn = Math.min(N, Math.round(0.015 * fsAudio));
    const fadeOut = Math.min(N, Math.round(0.025 * fsAudio));
    for (let n = 0; n < fadeIn; n++) {
      data[n] *= n / Math.max(1, fadeIn);
    }
    for (let n = 0; n < fadeOut; n++) {
      const idx = N - 1 - n;
      if (idx >= 0) data[idx] *= n / Math.max(1, fadeOut);
    }

    return buffer;
  }

  function playFromAnalysis(analysisData, chirpletVisibility) {
    if (!analysisData || !analysisData.result || !analysisData.result.chirplets) return;
    console.log('Playing from analysis... freqMode=', Audio.exportFrequencyMode);
    const ctx = getCtx();
    initAudioEngine();

    // Resume audio (required by browsers)
    if (ctx.state !== 'running') {
      try { ctx.resume(); } catch(e) {}
    }
    console.log('Audio context state:', ctx.state);

    // Stop any existing schedule
    stopAll();

    const fs = analysisData.sampling_frequency;
    const isCombined = analysisData.analysis_type === 'multi_sample_combined';

    // Detect whether time_center_seconds are global or local for non-combined files
    const localDur = analysisData.num_samples / fs;
    let maxTc = 0;
    if (analysisData.result.chirplets && analysisData.result.chirplets.length > 0) {
      for (let i = 0; i < analysisData.result.chirplets.length; i++) {
        const tcs = Number(analysisData.result.chirplets[i].time_center_seconds) || 0;
        if (tcs > maxTc) maxTc = tcs;
      }
    }
    const timesAreGlobal = isCombined || (maxTc > localDur * 1.25);
    const segmentStartSec = timesAreGlobal ? (analysisData.start_sample / fs) : 0;
    const segmentEndSec = segmentStartSec + (analysisData.num_samples / fs);
    console.log(`Timebase detection: isCombined=${isCombined}, maxTc=${maxTc.toFixed(6)}, localDur=${localDur.toFixed(6)}, timesAreGlobal=${timesAreGlobal}. Segment ${segmentStartSec.toFixed(6)}..${segmentEndSec.toFixed(6)}`);
    const k = Audio.coverageK;

    const all = analysisData.result.chirplets;
    console.log(`Found ${all.length} chirplets`);

    // Select voices to play
    const selected = [];
    for (let i = 0; i < all.length; i++) {
      const ch = all[i];
      if (Audio.playVisibleOnly && !chirpletVisibility[i]) continue;
      console.log(`Selected chirplet ${i}`);
      selected.push({ ch, i });
    }
    if (selected.length === 0) return;

    // Power-fair normalization from coefficients
    let sumCoeff2 = 0;
    for (const { ch } of selected) sumCoeff2 += (ch.coefficient * ch.coefficient);
    const coeffNorm = Math.sqrt(Math.max(1e-12, sumCoeff2));

    const base = ctx.currentTime + 0.2; // scheduling offset

    for (const { ch, i } of selected) {
      console.log(`Playing chirplet ${i}...`);
      const Dt = ch.duration_ms / 1000.0;
      const tc = ch.time_center_seconds; // already local for single, global for combined
      let t0 = tc - k * Dt;
      let t1 = tc + k * Dt;

      // Clamp to segment only when times are local; for global-time singles, don't clamp
      if (!timesAreGlobal) {
        t0 = Math.max(t0, segmentStartSec);
        t1 = Math.min(t1, segmentEndSec);
      }

      // Clamp to original window only for combined JSONs
      if (isCombined && typeof ch.__windowStart === 'number' && typeof ch.__windowSize === 'number') {
        const ws = ch.__windowStart / fs;
        const we = ws + ch.__windowSize / fs;
        t0 = Math.max(t0, ws);
        t1 = Math.min(t1, we);
      }
      console.log(`Chirplet ${i} time range: ${t0.toFixed(6)} to ${t1.toFixed(6)} (segment ${segmentStartSec.toFixed(6)}..${segmentEndSec.toFixed(6)}; combined=${isCombined})`);

      if (t1 <= t0) {
        console.warn(`Skipping chirplet ${i}: empty window after clamping.`);
        continue;
      }

      const buffer = makeChirpletBuffer(ch, t0, t1, fs, Audio.pitchScale, undefined, Audio.exportFrequencyMode);
      console.log(`Chirplet ${i} buffer duration: ${buffer.duration}`);
      console.log(`Chirplet ${i} buffer sample rate: ${buffer.sampleRate}`);
      console.log(`Chirplet ${i} buffer length: ${buffer.length}`);
      const source = ctx.createBufferSource();
      source.buffer = buffer;

      // Per-voice gain from coefficient normalization and master volume
      const voiceGain = ctx.createGain();
      const amp = (Math.abs(ch.coefficient) / coeffNorm) * Audio.mixGain; // unit-less, scaled by master
      voiceGain.gain.setValueAtTime(amp, ctx.currentTime);

      source.connect(voiceGain);
      voiceGain.connect(Audio.masterGain);

      const startOffset = Math.max(0, (t0 - segmentStartSec));
      const when = base + startOffset;
      try {
        source.start(when);
        source.stop(when + buffer.duration);
      } catch(e) {
        console.error('Audio play error:', e);
        // Some browsers throw if scheduled in the past; skip in that case
        continue;
      }
      console.log(`Scheduled chirplet ${i} from ${t0} to ${t1}`);
      Audio.sources.push({ source, gain: voiceGain, when, stopAt: when + buffer.duration });
    }
    console.log(`Scheduled ${selected.length} chirplets`);
    Audio.isPlaying = true;
  }

  // Expose API
  global.chirpletAudio = {
    init: initAudioEngine,
    setMasterVolume,
    setLimiterEnabled,
    setPitchScale,
    setCoverageK,
    setMixGain,
    setExportFrequencyMode,
    setPlayVisibleOnly,
    playFromAnalysis,
    stopAll,
    exportWav,
  };

})(window);
