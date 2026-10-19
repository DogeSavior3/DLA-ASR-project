import argparse
from pathlib import Path

from src.metrics.utils import calc_cer, calc_wer
from src.text_encoder.ctc_text_encoder import CTCTextEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("transcriptions", type=Path)
    args = parser.parse_args()

    files = sorted(args.predictions.glob("*.txt"))
    if not files:
        raise ValueError("No prediction .txt files found")

    cer, wer = 0.0, 0.0
    for path in files:
        prediction = CTCTextEncoder.normalize_text(
            path.read_text(encoding="utf-8")
        ).strip()
        target = CTCTextEncoder.normalize_text(
            (args.transcriptions / path.name).read_text(encoding="utf-8")
        ).strip()
        cer += calc_cer(target, prediction)
        wer += calc_wer(target, prediction)

    mean_cer = 100 * cer / len(files)
    mean_wer = 100 * wer / len(files)
    print("Mean utterance CER: %.2f%%" % mean_cer)
    print("Mean utterance WER: %.2f%%" % mean_wer)


if __name__ == "__main__":
    main()
