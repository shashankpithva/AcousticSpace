import { useEffect, useRef, useState } from "react"
import WaveSurfer from "wavesurfer.js"
import Timeline from "wavesurfer.js/dist/plugins/timeline.esm.js"

interface Props {
  audioUrl: string | null
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds)) return "0:00"
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m}:${s.toString().padStart(2, "0")}`
}

// Week 2: full Wavesurfer.js integration — waveform + timeline, playback
// controls, live time readout, loading state, and zoom.
export default function WaveformPanel({ audioUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)
  const [playing, setPlaying] = useState(false)
  const [ready, setReady] = useState(false)
  const [current, setCurrent] = useState(0)
  const [duration, setDuration] = useState(0)
  const [zoom, setZoom] = useState(0)

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return

    setReady(false)
    setPlaying(false)
    setCurrent(0)
    wsRef.current?.destroy()

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#7c9cff",
      progressColor: "#3b5bdb",
      cursorColor: "#e8ecf7",
      height: 96,
      barWidth: 2,
      barGap: 1,
      barRadius: 2,
      plugins: [Timeline.create()],
    })
    ws.load(audioUrl)
    ws.on("ready", () => {
      setReady(true)
      setDuration(ws.getDuration())
    })
    ws.on("timeupdate", (t: number) => setCurrent(t))
    ws.on("play", () => setPlaying(true))
    ws.on("pause", () => setPlaying(false))
    ws.on("finish", () => setPlaying(false))
    wsRef.current = ws

    return () => {
      ws.destroy()
      wsRef.current = null
    }
  }, [audioUrl])

  function handleZoom(value: number) {
    setZoom(value)
    wsRef.current?.zoom(value)
  }

  if (!audioUrl) {
    return <p className="muted">Upload a clip to see its waveform.</p>
  }

  return (
    <div className="waveform">
      {!ready && <p className="muted small">Loading waveform…</p>}
      <div ref={containerRef} />

      <div className="wave-controls">
        <button
          className="btn"
          disabled={!ready}
          onClick={() => wsRef.current?.playPause()}
        >
          {playing ? "⏸ Pause" : "▶️ Play"}
        </button>
        <span className="time">
          {formatTime(current)} / {formatTime(duration)}
        </span>
        <label className="zoom">
          Zoom
          <input
            type="range"
            min={0}
            max={200}
            value={zoom}
            disabled={!ready}
            onChange={(e) => handleZoom(Number(e.target.value))}
          />
        </label>
      </div>
    </div>
  )
}
