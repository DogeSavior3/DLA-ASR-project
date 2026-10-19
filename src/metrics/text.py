from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_cer, calc_wer


class TextCERMetric(BaseMetric):
    def __init__(self, text_encoder, prediction_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.prediction_key = prediction_key

    def __call__(self, text, **batch):
        return sum(
            calc_cer(self.text_encoder.normalize_text(target).strip(), prediction)
            for target, prediction in zip(text, batch[self.prediction_key])
        ) / len(text)


class TextWERMetric(BaseMetric):
    def __init__(self, text_encoder, prediction_key, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder
        self.prediction_key = prediction_key

    def __call__(self, text, **batch):
        return sum(
            calc_wer(self.text_encoder.normalize_text(target).strip(), prediction)
            for target, prediction in zip(text, batch[self.prediction_key])
        ) / len(text)
