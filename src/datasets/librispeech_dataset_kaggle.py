import json
import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset

URL_LINKS = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "dev-other": "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDatasetKaggle(BaseDataset):
    """Read attached LibriSpeech audio and cache indexes in a writable directory."""

    def __init__(self, part, data_dir, index_dir, *args, **kwargs):
        assert part in URL_LINKS or part == "train_all"

        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        if part == "train_all":
            index = sum(
                [
                    self._get_or_load_index(part)
                    for part in URL_LINKS
                    if "train" in part
                ],
                [],
            )
        else:
            index = self._get_or_load_index(part)

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if not index_path.is_file():
            matches = list(self._index_dir.rglob(index_path.name))
            if len(matches) == 1:
                index_path = matches[0]
        if not index_path.is_file() and self._index_dir.is_relative_to("/kaggle/input"):
            raise FileNotFoundError(
                f"Missing index {part}_index.json under {self._index_dir}. "
                "Attach the indexes dataset and check index_dir."
            )
        if index_path.exists():
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            self._index_dir.mkdir(exist_ok=True, parents=True)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        audio_dir = self._data_dir / part
        nested_dir = audio_dir / "LibriSpeech" / part
        if nested_dir.is_dir():
            audio_dir = nested_dir
        for entry in index:
            old_path = Path(entry["path"])
            split_dir = next(
                parent for parent in old_path.parents if parent.name == part
            )
            entry["path"] = str(audio_dir / old_path.relative_to(split_dir))
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            raise FileNotFoundError(f"LibriSpeech split not found: {split_dir}")

        flac_dirs = set()
        for dirpath, dirnames, filenames in os.walk(str(split_dir)):
            if any([f.endswith(".flac") for f in filenames]):
                flac_dirs.add(dirpath)
        for flac_dir in flac_dirs:
            flac_dir = Path(flac_dir)
            trans_path = list(flac_dir.glob("*.trans.txt"))[0]
            with trans_path.open() as f:
                total = sum(1 for _ in f)
                f.seek(0)
                for line in tqdm(
                    f, total=total, desc=f"Indexing LibriSpeech: {part}", unit="file"
                ):
                    f_id = line.split()[0]
                    f_text = " ".join(line.split()[1:]).strip()
                    flac_path = flac_dir / f"{f_id}.flac"
                    t_info = torchaudio.info(str(flac_path))
                    length = t_info.num_frames / t_info.sample_rate
                    index.append(
                        {
                            "path": str(flac_path.absolute().resolve()),
                            "text": f_text.lower(),
                            "audio_len": length,
                        }
                    )
        return index
