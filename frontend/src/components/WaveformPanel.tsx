import { useEffect, useRef } from "react"
import WaveSurfer from "wavesurfer.js"

interface Props {
  audioUrl: string | null
}

export default function WaveformPanel({ audioUrl }: Props) {

  const containerRef = useRef<HTMLDivElement | null>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)

  useEffect(() => {

    if (!containerRef.current || !audioUrl) return

    wavesurferRef.current = WaveSurfer.create({
      container: containerRef.current,
      height: 120,
      waveColor: "#4f46e5",
      progressColor: "#ef4444",
      cursorColor: "#111827",
      barWidth: 3,
      responsive: true,
    })

    wavesurferRef.current.load(audioUrl)


    return () => {
      wavesurferRef.current?.destroy()
    }

  }, [audioUrl])


  function togglePlay(){

    wavesurferRef.current?.playPause()

  }


  if (!audioUrl) {
    return (
      <p className="muted">
        Upload audio to view waveform
      </p>
    )
  }


  return (

    <div>

      <div ref={containerRef}></div>

      <button onClick={togglePlay}>
        ▶️ Play / Pause
      </button>

    </div>

  )
}