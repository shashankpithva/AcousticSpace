import { useEffect, useState } from "react"
import AudioUpload from "./components/AudioUpload"
import WaveformPanel from "./components/WaveformPanel"
import SpectrogramPanel from "./components/SpectrogramPanel"
import ResultsPanel from "./components/ResultsPanel"
import { analyzeAudio, checkHealth, type AnalyzeResponse } from "./api"

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [result, setResult] = useState<AnalyzeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiOnline, setApiOnline] = useState<boolean | null>(null)

  useEffect(() => {
    checkHealth().then(setApiOnline)
  }, [])

  async function handleFile(selected: File) {
    setFile(selected)
    setResult(null)
    setError(null)
    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl(URL.createObjectURL(selected))

    setLoading(true)
    try {
      const res = await analyzeAudio(selected)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">🎧</span>
          <div>
            <h1>AcousticSpace</h1>
            <p>Deepfake Detection via Room Impulse Response (RIR)</p>
          </div>
        </div>
        <span className={`status ${apiOnline ? "online" : "offline"}`}>
          {apiOnline === null ? "checking…" : apiOnline ? "API online" : "API offline"}
        </span>
      </header>

      <main className="grid">
        <section className="panel">
          <h2>1 · Upload Audio</h2>
          <AudioUpload onFile={handleFile} />
          {file && <p className="filename">{file.name}</p>}
        </section>

        <section className="panel">
          <h2>2 · Waveform</h2>
          <WaveformPanel audioUrl={audioUrl} />
        </section>

        <section className="panel">
          <h2>3 · Spectrogram</h2>
          <SpectrogramPanel melShape={result?.mel_shape ?? null} />
        </section>

        <section className="panel results">
          <h2>4 · Results</h2>
          {loading && <p className="muted">Analyzing…</p>}
          {error && <p className="error">{error}</p>}
          <ResultsPanel result={result} />
        </section>
      </main>

      <footer className="footer">
        Week 1 scaffold · feature extraction active · classifier lands Week 2
      </footer>
    </div>
  )
}
