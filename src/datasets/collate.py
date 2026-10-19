import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    spectrograms = [item["spectrogram"] for item in dataset_items]
    texts_encoded = [item["text_encoded"].squeeze(0).long() for item in dataset_items]

    spectrogram_length = torch.tensor([i.shape[-1] for i in spectrograms])
    text_encoded_length = torch.tensor([i.shape[-1] for i in texts_encoded])

    max_spectrogram_length = max(i.shape[-1] for i in spectrograms)
    spectrograms = [
        torch.nn.functional.pad(i, (0, max_spectrogram_length - i.shape[-1]), value=0)
        for i in spectrograms
    ]
    batch_spectrograms = torch.cat(spectrograms, dim=0)
    batch_texts = torch.nn.utils.rnn.pad_sequence(
        texts_encoded, batch_first=True, padding_value=0
    )

    return {
        "spectrogram": batch_spectrograms,
        "spectrogram_length": spectrogram_length,
        "text_encoded": batch_texts,
        "text_encoded_length": text_encoded_length,
        "text": [item["text"] for item in dataset_items],
        "has_transcription": [
            item["has_transcription"] if "has_transcription" in item else True
            for item in dataset_items
        ],
        "audio_path": [item["audio_path"] for item in dataset_items],
        "audio": [item["audio"] for item in dataset_items],
    }
