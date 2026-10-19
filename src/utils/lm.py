from pathlib import Path

import torch
from ctc_beam_decoder import build_ctcdecoder


class CTCBeamSearchLM:
    def __init__(
        self,
        text_encoder,
        lm_path=None,
        lexicon_path=None,
        beam_size=20,
        lm_weight=0.5,
        word_score=1.0,
    ):
        self.text_encoder = text_encoder
        self.beam_size = beam_size
        if lm_path is None:
            directory = Path("data/lm/librispeech-3-gram")
            directory.mkdir(parents=True, exist_ok=True)
            for filename in ["lm.bin", "lexicon.txt"]:
                path = directory / filename
                if not path.is_file():
                    torch.hub.download_url_to_file(
                        "https://download.pytorch.org/torchaudio/"
                        f"decoder-assets/librispeech-3-gram/{filename}",
                        str(path),
                    )
            lm_path = directory / "lm.bin"
            lexicon_path = directory / "lexicon.txt"

        unigrams = None
        if lexicon_path is not None:
            with Path(lexicon_path).open(encoding="utf-8") as source:
                unigrams = [line.split()[0] for line in source if line.strip()]

        tokens = [text_encoder.ind2char[i] for i in range(len(text_encoder))]
        tokens[text_encoder.EMPTY_IND] = ""
        self.decoder = build_ctcdecoder(
            tokens,
            kenlm_model_path=str(lm_path),
            unigrams=unigrams,
            alpha=lm_weight,
            beta=word_score,
        )

    def __call__(self, log_probs):
        if isinstance(log_probs, torch.Tensor):
            log_probs = log_probs.detach().float().cpu().numpy()
        text = self.decoder.decode(log_probs, beam_width=self.beam_size)
        return self.text_encoder.normalize_text(text).strip()
