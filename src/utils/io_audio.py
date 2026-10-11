import soundfile as sf
import torch


def load_audio(path: str):
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.T.copy())
    return waveform, sr
