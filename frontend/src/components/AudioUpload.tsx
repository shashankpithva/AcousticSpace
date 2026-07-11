import { useCallback, useRef, useState } from "react"

interface Props {
  onFile: (file: File) => void
}

const ACCEPT = ".wav,.mp3,.flac"

export default function AudioUpload({ onFile }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = useCallback(
    (files: FileList | null) => {
      if (files && files.length > 0) onFile(files[0])
    },
    [onFile],
  )

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault()
        setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault()
        setDragging(false)
        handleFiles(e.dataTransfer.files)
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <span className="drop-icon">⬆️</span>
      <p>
        <strong>Drag &amp; drop</strong> an audio clip, or <span className="link">browse</span>
      </p>
      <p className="muted small">Supports .wav · .mp3 · .flac</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  )
}
