import type { AnalyzeResponse } from "../api"

interface Props {
  segments: AnalyzeResponse["suspicious_segments"]
}

export default function SegmentsPanel({ segments }: Props) {

  if (!segments || segments.length === 0) {
    return <p className="muted">No suspicious segments.</p>
  }

  return (
    <div className="segments">

      {segments.map((seg, index) => (

        <div className="segment" key={index}>

          <strong>
            {seg.start}s - {seg.end}s
          </strong>

          <div className="bar">
            <div
              className="fill"
              style={{
                width: `${seg.fake_probability * 100}%`
              }}
            />
          </div>

          <p>
            Fake probability:
            {" "}
            {(seg.fake_probability * 100).toFixed(1)}%
          </p>

        </div>

      ))}

    </div>
  )
}