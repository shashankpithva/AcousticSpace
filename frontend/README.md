# AcousticSpace — Frontend (Week 1)

React + TypeScript + Vite analyst dashboard scaffold.

## Setup

```bash
npm install
npm run dev      # http://localhost:5173
```

The dev server proxies `/api/*` to the FastAPI backend on `http://localhost:8000`,
so start the backend first.

## Layout

- `src/components/AudioUpload.tsx` — drag & drop / browse upload (.wav .mp3 .flac).
- `src/components/WaveformPanel.tsx` — Wavesurfer.js waveform + playback.
- `src/components/SpectrogramPanel.tsx` — spectrogram placeholder (real image in Week 2/3).
- `src/components/ResultsPanel.tsx` — prediction, confidence, and the four key indicators.
- `src/api.ts` — typed client matching the backend response contract.

## Build

```bash
npm run build && npm run preview
```
