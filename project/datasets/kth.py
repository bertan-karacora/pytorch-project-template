from pathlib import Path
import urllib.request
import zipfile

import numpy as np
import torch
import torchvision as tv


class KTH(torch.utils.data.Dataset):
    filename_sequences = "00sequences.txt"
    indices_persons_splits = {
        "train": np.asarray([11, 12, 13, 14, 15, 16, 17, 18]),
        "validation": np.asarray([19, 20, 21, 23, 24, 25, 1, 4]),
        "test": np.asarray([22, 2, 3, 5, 6, 7, 8, 9, 10]),
    }
    labelset = np.asarray(["boxing", "handclapping", "handwaving", "jogging", "running", "walking"])
    scenarios = np.asarray(["outdoors", "outdoors with scale variation", "outdoors with different clothes", "indoors"])
    url = "https://www.csc.kth.se/cvap/actions/"

    def __init__(self, path, split="train", transform=None, transform_target=None, use_download=False, len_sequence=15):
        self.descriptors = None
        self.sequences = None
        self.len_sequence = len_sequence
        self.path = Path(path)
        self.split = split
        self.transform = transform
        self.transform_target = transform_target
        self.use_download = use_download

        self._init()

    def _init(self):
        self._init_descriptors()
        self._init_sequences()

    def _init_descriptors(self):
        # Example row from sequences file:
        # person01_boxing_d1		frames	1-95, 96-185, 186-245, 246-360

        # Skip header rows
        skip_rows = 21
        self.descriptors = {}
        with open(self.path / self.filename_sequences, "r") as file_sequences:
            for _ in range(skip_rows):
                next(file_sequences)
            for descriptor_sequence in file_sequences:
                if "missing" in descriptor_sequence:
                    continue

                descriptor_sequence_split = descriptor_sequence.split()
                if descriptor_sequence_split:
                    self.descriptors[descriptor_sequence_split[0]] = [
                        tuple(map(int, descriptor_range.replace(",", "").split("-"))) for descriptor_range in descriptor_sequence_split[2:]
                    ]

    def _init_sequences(self):
        # Treat subsequences as independent
        # Drop incomplete subsequences if needed
        # Filter out subsequences not fully contained in original sequences
        # Allow subsequences across two or more original sequences
        self.sequences = []
        for l, label in enumerate(self.labelset):
            for index_person in self.indices_persons_splits[self.split]:
                for s, scenario in enumerate(self.scenarios):
                    dir_scenario = f"person{index_person:02d}_{label}_d{s+1}"
                    if dir_scenario not in self.descriptors:
                        continue
                    ranges = self.unionize_ranges(self.descriptors[dir_scenario])
                    path_scenario = self.path / label / dir_scenario
                    paths_frames = sorted(path_scenario.iterdir())

                    # Nice
                    for i, sequence in enumerate(zip(*[iter(paths_frames)] * self.len_sequence)):
                        # Not so nice anymore
                        for range in ranges:
                            if i * self.len_sequence >= range[0] and (i + 1) * self.len_sequence <= range[1]:
                                sequence = [str(p.relative_to(self.path)) for p in sequence]
                                self.sequences += [(sequence, l)]

    def unionize_ranges(self, ranges):
        # See: https://stackoverflow.com/questions/15273693/union-of-multiple-ranges
        ranges_unionized = []
        for begin, end in sorted(ranges):
            if ranges_unionized and ranges_unionized[-1][1] >= begin - 1:
                ranges_unionized[-1][1] = max(ranges_unionized[-1][1], end)
            else:
                ranges_unionized += [[begin, end]]
        return ranges_unionized

    def __len__(self):
        length = len(self.sequences)
        return length

    def __getitem__(self, index):
        paths_subsamples_relative, target = self.sequences[index]

        subsamples = []
        for path_subsample_relative in paths_subsamples_relative:
            path_subsample = self.path / path_subsample_relative
            subsample = tv.io.read_image(path_subsample, mode=tv.io.ImageReadMode.GRAY)
            subsamples += [subsample]
        sample = torch.stack(subsamples)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.transform_target is not None:
            target = self.transform_target(target)

        return sample, target

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of samples: {self.__len__()}
    Path: {self.path}
    Split: {self.split}
    Transform of samples: {self.transform}
    Transform of targets: {self.transform_target}"""
        return s

    def download(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if not any(self.path.iterdir()):
            urllib.request.urlretrieve(self.url, self.path / self.filename_sequences)
            for label in self.labelset:
                path_zip = self.path / f"{label}.zip"
                path_videos = self.path / label
                urllib.request.urlretrieve(self.url, path_zip)
                with zipfile.ZipFile(path_zip, "r") as file_zip:
                    file_zip.extractall(path_videos)
                path_zip.unlink()
