# Based on seminar materials
import editdistance

# Don't forget to support cases when target_text == ''


def calc_cer(target_text: str, pred_text: str) -> float:
    if not target_text:
        return 0.0 if not pred_text else 1.0

    return editdistance.eval(target_text, pred_text) / len(target_text)


def calc_wer(target_text: str, pred_text: str) -> float:
    target = target_text.split()
    pred = pred_text.split()

    if not target:
        return 0.0 if not pred else 1.0
    return editdistance.eval(target, pred) / len(target)
