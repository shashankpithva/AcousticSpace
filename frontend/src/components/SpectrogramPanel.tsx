interface Props {
  melShape: number[] | null
}

// Week 1 placeholder. In Week 2/3 the backend will return the actual
// spectrogram image (or raw matrix) to render here.
export default function SpectrogramPanel({ melShape }: Props) {
  return (
    <div className="spectrogram">
      {melShape ? (
        <>
          <div className="spectro-placeholder" aria-hidden />
          <p className="muted small">
            Mel spectrogram computed: {melShape[0]} mels × {melShape[1]} frames
          </p>
        </>
      ) : (
        <p className="muted">Spectrogram appears here after analysis.</p>
      )}
    </div>
  )
}
