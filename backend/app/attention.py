import numpy as np
import librosa


def generate_attention(path):

    y, sr = librosa.load(
        path,
        sr=16000
    )

    duration = librosa.get_duration(
        y=y,
        sr=sr
    )


    windows = int(duration)

    values=[]


    for i in range(windows):

        start=i*sr
        end=(i+1)*sr

        chunk=y[start:end]


        energy=float(
            np.mean(
                np.abs(chunk)
            )
        )


        values.append(
            round(
                min(energy*10,1),
                3
            )
        )


    return values