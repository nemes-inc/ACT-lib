# EEG ACT Analyzer JSON Format Specification

Version: 1.0
Generator: `eeg_act_analyzer` (C++) in this repository

This document specifies the JSON output formats produced by `eeg_act_analyzer.cpp`. These files are intended to be consumed by readers for machine learning pipelines and visualization tools.

The analyzer exports three related formats, distinguished by `analysis_type`:
- `single` – single, contiguous segment analysis with exact reconstruction arrays.
- `multi_sample` – per-window analysis payload (one JSON per window if emitted). Contains per-window reconstruction arrays.
- `multi_sample_combined` – consolidated multi-window analysis in one file with flattened chirplets (no global reconstruction arrays).

All numeric values are written using `std::fixed` with `std::setprecision(12)`.

---

## 1. Common Root Fields
Present in all formats.

- `source_file` (string)
  - Path or filename of the CSV source.
- `column_name` (string)
  - Name of the selected column.
- `column_index` (int)
  - 0-based index of the selected column in the CSV.
- `start_sample` (int)
  - Inclusive start sample (0-based) of the analyzed segment in the full column.
- `num_samples` (int)
  - Number of samples in the selected segment.
  - For `multi_sample_combined`: equals `end_sample - start_sample` (see §3.3). This uses `end_sample` as an exclusive bound.
- `sampling_frequency` (float, Hz)
  - Sampling rate used (e.g., 256.0 for Muse data).
- `param_ranges` (object) – parameter grid used to build the dictionary:
  - `tc_min`, `tc_max`, `tc_step` (samples)
  - `fc_min`, `fc_max`, `fc_step` (Hz)
  - `logDt_min`, `logDt_max`, `logDt_step` (unitless)
  - `c_min`, `c_max`, `c_step` (Hz/s)

Notes:
- Internally, duration parameter is `logDt` (unitless). The JSON stores per-chirplet `duration_ms = 1000 * exp(logDt)`.
- `tc_max` defaults to `selected_signal.size() - 1` if not explicitly set before dictionary creation.

---

## 2. Chirplet Parameterization
Each chirplet is encoded by the tuple `(tc, fc, logDt, c, coeff)` with the following meanings:

- `time_center_samples` (float)
  - Time center `tc` in samples. May be fractional due to optimization.
- `time_center_seconds` (float)
  - `tc / sampling_frequency`.
- `frequency_hz` (float)
  - Center frequency `fc` in Hz.
- `duration_ms` (float)
  - `1000 * exp(logDt)`. To recover `Dt` (seconds), use `Dt = duration_ms / 1000`.
- `chirp_rate_hz_per_s` (float)
  - Linear chirp rate `c` in Hz/s.
- `coefficient` (float)
  - Least-squares amplitude of the chirplet atom (atoms are unit-energy; see below).

### 2.1 Chirplet Atom (real-valued) and Normalization
Given sampling rate `FS`, length `L` (dictionary/window length), and parameters `(tc, fc, Dt, c)`:

- Convert `tc_samples` to seconds: `t_c = tc_samples / FS`.
- For sample index `n = 0..L-1`, with `t = n / FS` and `t_rel = t - t_c`:

```
g[n] = exp(-0.5 * (t_rel / Dt)^2) * cos( 2π * ( c * t_rel^2 + fc * t_rel ) )
```

- Unit-energy normalization (performed in C++):
  - Let `E = sum_n g[n]^2`. If `E > 0`, set `g[n] <- g[n] / sqrt(E)` so `sum_n g[n]^2 = 1`.
- Instantaneous frequency (for diagnostics/visualization):

```
f_inst(t) = fc + 2 * c * (t - t_c)
```

### 2.2 Reconstruction With Coefficients
- Single window: `approx[n] = sum_i coeff[i] * g_i[n]`.
- Residual: `residue[n] = signal[n] - approx[n]`.

---

## 3. Formats

### 3.1 `analysis_type = "single"`
Single segment analysis; includes exact reconstruction arrays.

Additional root fields:
- `num_chirplets` (int)
- `residual_threshold` (float)

Result object:
- `result.error` (float)
  - Final residual L2 norm.
- `result.chirplets` (array)
  - Each chirplet object has fields described in §2. Example:

```
{
  "index": 1,                                 // 1-based index within this analysis only
  "time_center_samples": 1234.567890123456,
  "time_center_seconds": 4.8225,
  "frequency_hz": 10.125,
  "duration_ms": 512.000000000000,
  "chirp_rate_hz_per_s": -3.500000000000,
  "coefficient": 42.000000000000
}
```

- `result.approx` (array[float], length = `num_samples`)
- `result.residue` (array[float], length = `num_samples`)

### 3.2 `analysis_type = "multi_sample"`
Per-window analysis payload (one JSON per window). This format is included here for completeness.

Additional root fields:
- `num_chirps` (int)
- `end_sample` (int, exclusive upper bound for the overall sweep)
- `overlap` (int, samples)
- `window_size` (int, samples) – dictionary length `L`
- `window_start` (int, sample index in the full column)

Result object:
- `result.error` (float)
- `result.chirplets` (array)
  - `time_center_samples` are local to this window (i.e., 0..`window_size-1`).
  - To convert to global sample index, add `window_start`.
- `result.approx` (array[float], length = `window_size`)
- `result.residue` (array[float], length = `window_size`)

### 3.3 `analysis_type = "multi_sample_combined"`
Consolidated output for many windows in a single file. Optimized for visualization and further processing. No global `approx`/`residue` arrays are included.

Additional root fields:
- `num_chirps_per_window` (int)
- `end_sample` (int, exclusive upper bound)
- `overlap` (int, samples)
- `window_size` (int, samples) – dictionary length `L`
- `window_starts` (array[int]) – starting sample of each processed window

Result object:
- `result.error_per_window` (array[float], aligned with `window_starts`)
- `result.chirplets` (array)
  - Flattened across windows.
  - `index` is 1-based per-window and thus repeats across windows (do not treat as globally unique).
  - `time_center_samples` and `time_center_seconds` are GLOBAL (each local `tc` shifted by its window start).
  - Other fields as defined in §2.

Example snippet:
```
{
  "analysis_type": "multi_sample_combined",
  "window_size": 512,
  "window_starts": [24139, 24651, 25163],
  "result": {
    "error_per_window": [123.45, 118.76, 119.02],
    "chirplets": [
      {
        "index": 1,
        "time_center_samples": 28936.000000000000,
        "time_center_seconds": 113.031250000000,
        "frequency_hz": 1.438992376428,
        "duration_ms": 800.114849294554,
        "chirp_rate_hz_per_s": -8.118683083639,
        "coefficient": 30.967766124659
      },
      { /* next chirplet */ }
    ]
  }
}
```

---

## 4. Reconstruction Guidance for Consumers

### 4.1 Single (`analysis_type = "single"`)
- Preferred: use `result.approx` and `result.residue` directly.
- Or recompute `approx` via §2.2 from chirplets.

### 4.2 Multi-window (`analysis_type = "multi_sample"`)
- Per-window `result.approx` and `result.residue` are provided.
- To build a global reconstruction:
  1. Initialize a global buffer of length `num_samples` to 0.
  2. For each window:
     - Add the window `approx` into the global buffer at offset `window_start - start_sample`.
  3. In overlapped regions, either sum or average by the coverage count. A common strategy is simple averaging:

```
weights = zeros(num_samples)
for w in windows:
    s = window_start[w] - start_sample
    e = s + window_size
    global[s:e] += approx_w
    weights[s:e] += 1
for i in range(num_samples):
    if weights[i] > 0:
        global[i] /= weights[i]
```

### 4.3 Combined (`analysis_type = "multi_sample_combined"`)
- No global `approx`/`residue` arrays are included. Reconstruct from chirplets:
  - For each chirplet belonging to a window starting at `wstart`:
    1. Create the unit-energy atom `g_i[n]` of length `window_size` using §2.1.
    2. Form `component_i[n] = coeff_i * g_i[n]`.
    3. Place into the global array at indices `[wstart - start_sample, wstart - start_sample + window_size)` (zero outside the window).
  - Sum all components.
  - Optionally average in overlapped regions by coverage counts as above.

Notes:
- Because atoms are normalized over the window length used by the dictionary, coefficients represent amplitudes consistent with that window. When merging windows, coverage-weighted averaging helps avoid double counting in overlaps.

---

## 5. Preprocessing Notes (Reader Expectations)
- The analyzer removes NaNs in the selected segment and subtracts the mean (DC offset) before analysis.
- `end_sample` is treated as an exclusive upper bound in window iteration.
- Some parameters (e.g., `time_center_samples`) can be fractional; consumers should not assume integers.
- All floats are written with fixed-point precision (12 decimals). Use double-precision parsing.

---

## 6. Minimal Schemas

### 6.1 Common (pseudo-schema)
```
root: {
  source_file: string,
  column_name: string,
  column_index: int,
  start_sample: int,
  num_samples: int,
  sampling_frequency: float,
  param_ranges: {
    tc_min: float, tc_max: float, tc_step: float,
    fc_min: float, fc_max: float, fc_step: float,
    logDt_min: float, logDt_max: float, logDt_step: float,
    c_min: float, c_max: float, c_step: float
  },
  analysis_type: "single" | "multi_sample" | "multi_sample_combined",
  ... (type-specific fields)
}
```

### 6.2 Chirplet object
```
chirplet: {
  index: int, // 1-based within window/analysis; NOT globally unique in combined files
  time_center_samples: float,
  time_center_seconds: float,
  frequency_hz: float,
  duration_ms: float,
  chirp_rate_hz_per_s: float,
  coefficient: float
}
```

---

## 7. Backward/Forward Compatibility
- New fields may be added in future versions. Readers should ignore unknown fields.
- `result.approx`/`result.residue` are guaranteed for `single` and `multi_sample` but omitted for `multi_sample_combined` to keep files compact.
- The per-window `chirplet.index` repeats across windows in `multi_sample_combined`; do not rely on it for identity. Use array position or derive a composite key `(window_index, per_window_index)` if needed.

---

## 8. References
- Implementation: `eeg_act_analyzer.cpp` – functions `save_analysis_to_json(...)` and `save_multiwindow_combined_to_json(...)`.
- Atom definition: `ACT::g(...)` in `ACT.cpp`.
