// Typed client for the AcousticSpace backend.
// The API base is proxied through Vite ('/api' -> http://localhost:8000).

export interface ReverbFeatures {
  rt60_estimate_s: number
  drr_db: number
  spectral_flatness: number
  spectral_centroid_hz: number
  zero_crossing_rate: number
}

export interface KeyIndicators {
  rir_mismatch: string
  reverb_consistency: string
  breathing_pattern: string
  vocal_cadence: string
}

export interface AnalyzeResponse {
  filename: string
  duration_s: number
  prediction: string
  confidence: number
  model_stage: string
  reverb: ReverbFeatures
  key_indicators: KeyIndicators
  mel_shape: number[]
  notes: string
}

const API_BASE = "/api"

export async function analyzeAudio(file: File): Promise<AnalyzeResponse> {
  const form = new FormData()
  form.append("file", file)
  const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`Analyze failed (${res.status}): ${detail}`)
  }
  return (await res.json()) as AnalyzeResponse
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/health`)
    return res.ok
  } catch {
    return false
  }
}
