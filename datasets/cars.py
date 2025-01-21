import os
import torch
import torchvision.datasets as datasets


import pathlib
from typing import Callable, Optional, Any, Tuple

from PIL import Image
from torch.utils.data import BatchSampler

from torchvision.datasets.utils import download_and_extract_archive, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset
import numpy as np
from itertools import combinations


class PytorchStanfordCars(VisionDataset):
    """`Stanford Cars <https://ai.stanford.edu/~jkrause/cars/car_dataset.html>`_ Dataset

    The Cars dataset contains 16,185 images of 196 classes of cars. The data is
    split into 8,144 training images and 8,041 testing images, where each class
    has been split roughly in a 50-50 split

    .. note::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of dataset
        split (string, optional): The dataset split, supports ``"train"`` (default) or ``"test"``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:

        try:
            import scipy.io as sio
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transform=transform, target_transform=target_transform)

        self._split = verify_str_arg(split, "split", ("train", "test"))
        self._base_folder = pathlib.Path(root) / "stanford_cars"
        devkit = self._base_folder / "devkit"

        if self._split == "train":
            self._annotations_mat_path = devkit / "cars_train_annos.mat"
            self._images_base_path = self._base_folder / "cars_train"
        else:
            self._annotations_mat_path = devkit / "cars_test_annos_withlabels.mat"
            self._images_base_path = self._base_folder / "cars_test"

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        self._samples = [
            (
                str(self._images_base_path / annotation["fname"]),
                annotation["class"] - 1,  # Original target mapping  starts from 1, hence -1
            )
            for annotation in sio.loadmat(self._annotations_mat_path, squeeze_me=True)["annotations"]
        ]

        self.classes = sio.loadmat(str(devkit / "cars_meta.mat"), squeeze_me=True)["class_names"].tolist()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Returns pil_image and class_id for given index"""
        image_path, target = self._samples[idx]
        pil_image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            pil_image = self.transform(pil_image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return pil_image, target


    def download(self) -> None:
        if self._check_exists():
            return

        download_and_extract_archive(
            url="https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz",
            download_root=str(self._base_folder),
            md5="c3b158d763b6e2245038c8ad08e45376",
        )
        if self._split == "train":
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_train.tgz",
                download_root=str(self._base_folder),
                md5="065e5b463ae28d29e77c1b4b166cfe61",
            )
        else:
            download_and_extract_archive(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test.tgz",
                download_root=str(self._base_folder),
                md5="4ce7ebf6a94d07f1952d94dd34c4d501",
            )
            download_url(
                url="https://ai.stanford.edu/~jkrause/car196/cars_test_annos_withlabels.mat",
                root=str(self._base_folder),
                md5="b0a2b23655a3edd16d84508592a98d10",
            )

    def _check_exists(self) -> bool:
        if not (self._base_folder / "devkit").is_dir():
            return False

        return self._annotations_mat_path.exists() and self._images_base_path.is_dir()


class OneClassBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.class_indices = self._group_by_class()

    def _group_by_class(self):
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def __iter__(self):
        for class_indices in self.class_indices.values():
            # Split indices of the same class into batches
            for i in range(0, len(class_indices), self.batch_size):
                yield class_indices[i:i + self.batch_size]

    def __len__(self):
        return sum((len(indices) + self.batch_size - 1) // self.batch_size
                   for indices in self.class_indices.values())


class TwoClassBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.class_indices = self._group_by_class()
        self.class_pairs = self._generate_class_pairs()

    def _group_by_class(self):
        class_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        return class_indices

    def _generate_class_pairs(self):
        # Generate all possible pairs of classes
        return list(combinations(self.class_indices.keys(), 2))

    def __iter__(self):
        for class1, class2 in self.class_pairs:
            indices1 = self.class_indices[class1]
            indices2 = self.class_indices[class2]
            combined_indices = indices1 + indices2
            # Shuffle within the pair (optional)
            np.random.shuffle(combined_indices)

            # Split into batches
            for i in range(0, len(combined_indices), self.batch_size):
                yield combined_indices[i:i + self.batch_size]

    def __len__(self):
        # Approximate number of batches (depends on how the indices split into batches)
        total_batches = 0
        for class1, class2 in self.class_pairs:
            total_items = len(self.class_indices[class1]) + len(self.class_indices[class2])
            total_batches += (total_items + self.batch_size - 1) // self.batch_size
        return total_batches


class Cars:
    def __init__(self,
                 preprocess,
                 location=os.path.expanduser('~/data'),
                 batch_size=32,
                 num_workers=0,
                 ):
        # Data loading code

        self.train_dataset = PytorchStanfordCars(location, 'train', preprocess, download=False)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            num_workers=num_workers,
            shuffle=True,
            batch_size=batch_size,
        )

        self.test_dataset = PytorchStanfordCars(location, 'test', preprocess, download=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            num_workers=num_workers
        )
        self.test_loader_shuffle = torch.utils.data.DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers
        )

        labels = [l for _, l in self.test_dataset._samples]
        one_class_batch_sampler = OneClassBatchSampler(labels, batch_size)
        self.test_loader_one_class_batch = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_sampler=one_class_batch_sampler,
            num_workers=num_workers,
        )

        two_class_batch_sampler = TwoClassBatchSampler(labels, batch_size)
        self.test_loader_two_class_batch = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_sampler=two_class_batch_sampler,
            num_workers=num_workers,
        )

        idx_to_class = dict((v, k) for k, v in self.train_dataset.class_to_idx.items())
        self.classnames = [idx_to_class[i].replace(
            '_', ' ') for i in range(len(idx_to_class))]
