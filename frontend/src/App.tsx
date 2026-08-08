import { useEffect, useState } from "react"

import AudioUpload from "./components/AudioUpload"
import WaveformPanel from "./components/WaveformPanel"
import SpectrogramPanel from "./components/SpectrogramPanel"
import ResultsPanel from "./components/ResultsPanel"
import SuspiciousTimeline from "./components/SuspiciousTimeline"
import AttentionHeatmap from "./components/AttentionHeatmap"

import {
  analyzeAudio,
  checkHealth,
  getAttention,
  type AnalyzeResponse,
} from "./api"


export default function App() {

  const [file, setFile] = useState<File | null>(null)

  const [audioUrl, setAudioUrl] = useState<string | null>(null)

  const [result, setResult] =
    useState<AnalyzeResponse | null>(null)

  const [attention, setAttention] =
    useState<number[]>([])

  const [loading, setLoading] =
    useState(false)

  const [error, setError] =
    useState<string | null>(null)

  const [apiOnline, setApiOnline] =
    useState<boolean | null>(null)



  useEffect(() => {

    checkHealth()
      .then(setApiOnline)
      .catch(() => setApiOnline(false))

  }, [])



  async function handleFile(selected: File) {


    setFile(selected)

    setResult(null)

    setAttention([])

    setError(null)



    if (audioUrl) {

      URL.revokeObjectURL(audioUrl)

    }


    const url = URL.createObjectURL(selected)

    setAudioUrl(url)



    setLoading(true)



    try {


      const analysis =
        await analyzeAudio(selected)


      setResult(analysis)



      const att =
        await getAttention(selected)


      setAttention(
        att.attention
      )



    }

    catch (e) {


      setError(

        e instanceof Error
          ? e.message
          : String(e)

      )


    }

    finally {


      setLoading(false)


    }


  }



  return (

    <div className="app">


      <header className="topbar">


        <div className="brand">


          <span className="logo">
            🎧
          </span>


          <div>

            <h1>
              AcousticSpace
            </h1>


            <p>
              Deepfake Audio Detection using AST + RIR Analysis
            </p>

          </div>


        </div>



        <span
          className={
            `status ${
              apiOnline
                ? "online"
                : "offline"
            }`
          }
        >

          {
            apiOnline === null
              ? "checking..."
              : apiOnline
                ? "API online"
                : "API offline"
          }


        </span>


      </header>





      <main className="grid">



        <section className="panel">


          <h2>
            1 · Upload Audio
          </h2>


          <AudioUpload
            onFile={handleFile}
          />


          {
            file &&
            <p className="filename">
              {file.name}
            </p>
          }


        </section>





        <section className="panel">


          <h2>
            2 · Waveform
          </h2>


          <WaveformPanel
            audioUrl={audioUrl}
          />


        </section>





        <section className="panel">


          <h2>
            3 · Spectrogram
          </h2>


          <SpectrogramPanel
            melShape={
              result?.mel_shape ?? null
            }
          />


        </section>





        <section className="panel results">


          <h2>
            4 · Results
          </h2>



          {
            loading &&
            <p className="muted">
              Analyzing...
            </p>
          }



          {
            error &&
            <p className="error">
              {error}
            </p>
          }



          <ResultsPanel
            result={result}
          />


        </section>





        {
          result &&
          <section className="panel">


            <h2>
              5 · Suspicious Timeline
            </h2>


            <SuspiciousTimeline
              segments={
                result.suspicious_segments
              }
            />


          </section>
        }





        <section className="panel">


          <h2>
            6 · AST Attention Heatmap
          </h2>


          <AttentionHeatmap
            values={attention}
          />


        </section>



      </main>





      <footer className="footer">

        Week 3 AST deepfake detection pipeline active

      </footer>



    </div>

  )

}