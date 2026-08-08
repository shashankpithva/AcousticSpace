interface Props {
  values: number[]
}


export default function AttentionHeatmap({
  values,
}: Props) {


  if (!values || values.length === 0) {

    return (
      <p className="muted">
        No attention data.
      </p>
    )

  }


  return (

    <div className="attention">

      {values.map((v,index)=>(

        <div
          key={index}
          className="attention-box"
          style={{
            opacity:0.3 + v*0.7
          }}
        >

          {index}s

          <br/>

          {Math.round(v*100)}%

        </div>

      ))}


    </div>

  )
}