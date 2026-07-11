import type { AnalyzeResponse } from "../api"

interface Props {
  result: AnalyzeResponse | null
}

function Indicator({ label, value }: { label: string; value: string }) {
  const tone =
    value === "high" || value === "inconsistent" || value === "irregular"
      ? "bad"
      : value === "low" || value === "consistent" || value === "natural"
        ? "good"
        : value === "medium"
          ? "warn"
          : "neutral"
  return (
    <div className="indicator">
      <span className="indicator-label">{label}</span>
      <span className={`badge ${tone}`}>{value}</span>
    </div>
  )
}

export default function ResultsPanel({ result }: Props) {
  if (!result) return <p className="muted">No analysis yet.</p>

  const pct = Math.round(result.confidence * 100)
  return (
    <div className="results-body">
      <div className="verdict">
        <div>
          <span className="muted small">Prediction</span>
          <div className={`verdict-value ${result.prediction}`}>
            {result.prediction.toUpperCase()}
          </div>
        </div>
        <div>
          <span className="muted small">Confidence</span>
          <div className="verdict-value">{pct}%</div>
        </div>
      </div>

      <div className="indicators">
        <Indicator label="RIR Mismatch" value={result.key_indicators.rir_mismatch} />
        <Indicator label="Reverb Consistency" value={result.key_indicators.reverb_consistency} />
        <Indicator label="Breathing Pattern" value={result.key_indicators.breathing_pattern} />
        <Indicator label="Vocal Cadence" value={result.key_indicators.vocal_cadence} />
      </div>

      <details className="reverb-details">
        <summary>Extracted RIR / reverb features</summary>
        <ul>
          <li>RT60 estimate: {result.reverb.rt60_estimate_s} s</li>
          <li>Direct-to-reverberant ratio: {result.reverb.drr_db} dB</li>
          <li>Spectral flatness: {result.reverb.spectral_flatness}</li>
          <li>Spectral centroid: {result.reverb.spectral_centroid_hz} Hz</li>
          <li>Zero-crossing rate: {result.reverb.zero_crossing_rate}</li>
        </ul>
      </details>

      <p className="muted small">{result.notes}</p>
    </div>
  )
}
