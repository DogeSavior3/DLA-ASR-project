import sys
from pathlib import Path

import torch
from tqdm import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.beam_search import ctc_beam_search


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder
        self.lm_decoder = None
        if self.cfg_trainer.get("use_lm", False):
            from src.utils.lm import CTCBeamSearchLM

            self.lm_decoder = CTCBeamSearchLM(
                text_encoder=text_encoder,
                lm_path=self.cfg_trainer.lm_path,
                lexicon_path=self.cfg_trainer.lexicon_path,
                beam_size=self.cfg_trainer.lm_beam_size,
                lm_weight=self.cfg_trainer.lm_weight,
                word_score=self.cfg_trainer.word_score,
            )

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.metrics = {
                "inference": [
                    met
                    for met in metrics["inference"]
                    if getattr(met, "prediction_key", None) != "beam_lm_text"
                    or self.lm_decoder is not None
                ]
            }
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        batch_size = batch["log_probs"].shape[0]
        has_transcription = (
            batch["has_transcription"]
            if "has_transcription" in batch
            else [True] * batch_size
        )
        labeled_indices = [i for i, present in enumerate(has_transcription) if present]

        batch["argmax_text"] = []
        batch["beam_text"] = []
        if self.lm_decoder is not None:
            batch["beam_lm_text"] = []

        for i in range(batch_size):
            length = int(batch["log_probs_length"][i])
            log_probs = batch["log_probs"][i, :length].detach().float().cpu().numpy()
            argmax_text = self.text_encoder.ctc_decode(
                log_probs.argmax(axis=-1)
            ).strip()
            beam_text = ctc_beam_search(
                log_probs,
                beam_size=self.cfg_trainer.beam_size,
                ind2char=self.text_encoder.ind2char,
            )[0][0].strip()
            batch["argmax_text"].append(argmax_text)
            batch["beam_text"].append(beam_text)
            predictions = [("argmax", argmax_text), ("beam", beam_text)]
            if self.lm_decoder is not None:
                lm_text = self.lm_decoder(log_probs)
                batch["beam_lm_text"].append(lm_text)
                predictions.append(("beam_lm", lm_text))

            if self.save_path is not None:
                utterance_id = Path(batch["audio_path"][i]).stem
                for method, text in predictions:
                    output_dir = self.save_path / part / method
                    output_dir.mkdir(parents=True, exist_ok=True)
                    (output_dir / f"{utterance_id}.txt").write_text(
                        text.strip() + "\n", encoding="utf-8"
                    )

        if metrics is not None and labeled_indices:
            metric_batch = {
                "log_probs": batch["log_probs"][labeled_indices],
                "log_probs_length": batch["log_probs_length"]
                .detach()
                .cpu()[labeled_indices],
                "text": [batch["text"][i] for i in labeled_indices],
                "argmax_text": [batch["argmax_text"][i] for i in labeled_indices],
                "beam_text": [batch["beam_text"][i] for i in labeled_indices],
            }
            if self.lm_decoder is not None:
                metric_batch["beam_lm_text"] = [
                    batch["beam_lm_text"][i] for i in labeled_indices
                ]
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**metric_batch), n=len(labeled_indices))
            self._scored_samples += len(labeled_indices)

        return batch

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self._scored_samples = 0
        if self.evaluation_metrics is not None:
            self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
                file=sys.stdout,
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        if self.evaluation_metrics is not None and self._scored_samples:
            return self.evaluation_metrics.result()
        return {}
