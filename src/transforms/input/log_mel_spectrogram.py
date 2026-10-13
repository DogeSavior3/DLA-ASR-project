import torch
import torchaudio.transforms as T


class LogMelSpectrogram:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        win_length: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
    ) -> None:
        self.sample_rate = sample_rate
        self._mel_cache: dict[int, T.MelSpectrogram] = {}
        self.mel_kwargs = dict(
            n_fft=n_fft, win_length=win_length, hop_length=hop_length, n_mels=n_mels
        )

    def mel(self, sr: int) -> T.MelSpectrogram:
        if sr not in self._mel_cache:
            self._mel_cache[sr] = T.MelSpectrogram(sample_rate=sr, **self.mel_kwargs)
        return self._mel_cache[sr]

    @torch.no_grad()
    def log_mel(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # clamp: log(0) = -inf on silence
        return torch.log(self.mel(sr)(wav).clamp(min=1e-5))

    def __call__(self, wav: torch.Tensor, sr: int | None = None) -> torch.Tensor:
        return self.log_mel(wav, self.sample_rate if sr is None else sr)
