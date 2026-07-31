import type { AnalyzeResponse } from "../api"

interface Props {
  segments: AnalyzeResponse["suspicious_segments"]
}

export default function SuspiciousTimeline({
  segments,
}: Props) {

  if (!segments || segments.length === 0) {
    return (
      <p className="muted">
        No suspicious regions detected.
      </p>
    )
  }

  return (
    <div className="timeline">

      {segments.map((seg, index) => {

        const percent =
          Math.round(seg.fake_probability * 100)

        return (
          <div
            key={index}
            className="timeline-row"
          >

            <div className="time">
              {seg.start}s - {seg.end}s
            </div>

            <div className="bar">

              <div
                className="fill"
                style={{
                  width: `${percent}%`
                }}
              />

            </div>

            <span>
              {percent}% fake
            </span>

          </div>
        )
      })}

    </div>
  )
}