import { useEffect, useRef } from "react"
import WaveSurfer from "wavesurfer.js"

interface Props {
  audioUrl: string | null
}

// Renders the uploaded track as an interactive waveform using Wavesurfer.js.
// (Wavesurfer integration is a Week 2 item; the wiring is scaffolded here.)
export default function WaveformPanel({ audioUrl }: Props) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WaveSurfer | null>(null)

  useEffect(() => {
    if (!containerRef.current || !audioUrl) return

    wsRef.current?.destroy()
    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: "#7c9cff",
      progressColor: "#3b5bdb",
      height: 96,
      cursorColor: "#1c2b57",
    })
    ws.load(audioUrl)
    wsRef.current = ws

    return () => {
      ws.destroy()
      wsRef.current = null
    }
  }, [audioUrl])

  return (
    <div className="waveform">
      {!audioUrl && <p className="muted">Upload a clip to see its waveform.</p>}
      <div ref={containerRef} />
      {audioUrl && (
        <button className="btn" onClick={() => wsRef.current?.playPause()}>
          ▶️ Play / Pause
        </button>
      )}
    </div>
  )
}
