import json
import os
from pathlib import Path

import torchaudio
from tqdm import tqdm

from src.datasets.base_dataset import BaseDataset


class CommonVoiceDatasetKaggle(BaseDataset):
    def __init__(self, part, data_dir, index_dir, *args, **kwargs):
        self._data_dir = Path(data_dir)
        self._index_dir = Path(index_dir)
        parts = [part] if isinstance(part, str) else part
        index = []
        for split in parts:
            index.extend(self._get_or_load_index(split))

        super().__init__(index, *args, **kwargs)

    def _get_or_load_index(self, part):
        index_path = self._index_dir / f"{part}_index.json"
        if not index_path.is_file():
            matches = list(self._index_dir.rglob(index_path.name))
            if len(matches) == 1:
                index_path = matches[0]
        if index_path.exists() or self._index_dir.is_relative_to("/kaggle/input"):
            with index_path.open() as f:
                index = json.load(f)
        else:
            index = self._create_index(part)
            self._index_dir.mkdir(exist_ok=True, parents=True)
            with index_path.open("w") as f:
                json.dump(index, f, indent=2)
        for entry in index:
            entry["path"] = str(self._data_dir / part / Path(entry["path"]).name)
        return index

    def _create_index(self, part):
        index = []
        split_dir = self._data_dir / part
        if not split_dir.exists():
            raise FileNotFoundError(f"Common Voice split not found: {split_dir}")

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
                    f, total=total, desc=f"Indexing Common Voice: {part}", unit="file"
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
