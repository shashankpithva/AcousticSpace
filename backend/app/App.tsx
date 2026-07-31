import { useState } from "react";
import { analyzeAudio } from "./api";
import "./styles.css";


function App() {

  const [file, setFile] = useState<File | null>(null);

  const [result, setResult] = useState<any>(null);

  const [loading, setLoading] = useState(false);


  async function handleAnalyze() {

    if (!file) {
      alert("Select wav file first");
      return;
    }


    setLoading(true);


    try {

      const data = await analyzeAudio(file);

      setResult(data);

    }
    catch(err){

      alert("Error analyzing audio");

    }

    setLoading(false);
  }



  return (

    <div className="container">

      <h1>
        AcousticSpace
      </h1>


      <h2>
        Deepfake Audio Detection
      </h2>


      <input

        type="file"

        accept=".wav"

        onChange={
          e => setFile(
            e.target.files?.[0] || null
          )
        }

      />


      <button
        onClick={handleAnalyze}
      >

        {loading ? "Analyzing..." : "Analyze"}

      </button>



      {
        result &&

        <div className="result">


          <h2>
            Result
          </h2>


          <p>
            File: {result.filename}
          </p>


          <p>
            Prediction:
            <b>
              {result.prediction}
            </b>
          </p>


          <p>
            Confidence:
            {(result.confidence * 100).toFixed(2)}%
          </p>


          <h3>
            Acoustic Features
          </h3>


          <p>
            RT60:
            {result.reverb.rt60_estimate_s}
          </p>


          <p>
            Spectral Centroid:
            {result.reverb.spectral_centroid_hz}
          </p>


          <p>
            Breathing:
            {result.key_indicators.breathing_pattern}
          </p>



          <h3>
            Suspicious Segments
          </h3>


          {
            result.suspicious_segments.map(
              (seg:any,index:number)=>(

                <p key={index}>

                  {seg.start}s -
                  {seg.end}s

                  :
                  Fake probability
                  {(seg.fake_probability*100).toFixed(2)}%

                </p>

              )
            )
          }


        </div>

      }


    </div>

  );

}


export default App;