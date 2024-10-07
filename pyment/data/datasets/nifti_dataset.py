from __future__ import annotations

import logging
import os
import numpy as np
import pandas as pd

from typing import Dict

from .dataset import Dataset

logformat = '%(asctime)s - %(levelname)s - %(name)s: %(message)s'
logging.basicConfig(format=logformat, level=logging.INFO)
logger = logging.getLogger(__name__)


class NiftiDataset(Dataset):
    """
    Dataset class for handling NIfTI medical image files and associated labels.

    This class stores file paths to NIfTI images along with optional labels for
    each image, typically for use in machine learning applications. The dataset
    can be initialized directly with arrays of paths and labels, or loaded from
    a CSV file.

    Attributes:
        paths (np.ndarray): Array of file paths to NIfTI images.
        labels (Dict[str, np.ndarray], optional): Dictionary of labels where
            keys are label names and values are arrays of corresponding label
            data.
        target (str, optional): Specifies the target label or attribute that
            will be returned when accessing the `y` property. It can be:
                - A label column name from the dataset.
                - 'path': Returns the image file paths.
                - 'filename': Returns the filenames of the image files.
                - 'id': Returns the base name (without extension) of the
                  filenames.
                - None: Returns an array of `None` values.
    """

    def __init__(self, paths: np.ndarray, labels: Dict[str, np.ndarray] = None,
                 target: str = None) -> NiftiDataset:
        """
        Initializes the NiftiDataset with image paths and labels.

        Args:
            paths (np.ndarray): Array of file paths to NIfTI images.
            labels (Dict[str, np.ndarray], optional): Dictionary of labels with
                keys as label names and values as arrays of corresponding label
                data. Default is None.
            target (str, optional): The target variable to be returned by the
                `y` property. Default is None.

        Returns:
            NiftiDataset: The initialized dataset object.
        """
        self._paths = paths
        self._labels = labels
        self.target = target

    @classmethod
    def from_csv(cls,
                 csv_file: str,
                 images_col: str = 'path',
                 labels_cols: list[str] = None,
                 show_missing_warnings: bool = True,
                 **kwargs) -> NiftiDataset:
        """
        Creates a NiftiDataset from a CSV file.

        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            images_col (str, optional): The column name in the CSV that
                contains image paths. Default is 'path'.
            labels_cols (list[str], optional): A list of column names in the
                CSV that contain labels. If None, all columns except the image
                column will be used as labels. Default is None.
            show_missing_warnings (bool, optional): Whether to show warnings
                for missing files. Default is True.

        Returns:
            NiftiDataset: An instance of NiftiDataset with image paths and
                labels from the CSV file.

        Raises:
            AssertionError: If the `images_col` is not found in the CSV file.
        """
        df = pd.read_csv(csv_file)

        assert images_col in df.columns, f'{images_col} column is missing in the CSV file'

        if labels_cols is None:
            labels_cols = [col for col in df.columns if col != images_col]

        missing_files = df[~df[images_col].apply(os.path.exists)]

        if show_missing_warnings and not missing_files.empty:
            for _, row in missing_files.iterrows():
                logger.warning(f"Missing file: {row[images_col]}")

        df = df[df[images_col].apply(os.path.exists)]

        paths = df[images_col].values
        labels = {col: df[col].values for col in labels_cols}

        logger.debug((f'Creating {cls.__name__} with {len(df)} datapoints and '
                      f'labels {labels_cols}'))

        return cls(paths, labels, **kwargs)
    
    def shuffled(self):
        x = list(enumerate(self.paths))
        np.random.shuffle(x)
        indices, new = zip(*x)
        indices = list(indices)

        self._paths = list(new)
        for key in self._labels.keys():
            y = self._labels[key]
            self._labels[key] = y[indices]

        return self

    @property
    def variables(self):
        """
        Returns the list of available label variables.

        Returns:
            list: A list of keys (label names) from the labels dictionary.
        """
        if self._labels is None or len(self._labels) == 0:
            return []

        return list(self._labels.keys())

    @property
    def paths(self):
        """
        Returns the image paths.

        Returns:
            np.ndarray: Array of image file paths.
        """
        return self._paths

    @property
    def filenames(self):
        """
        Returns the base filenames of the images (excluding directories).

        Returns:
            list[str]: A list of base filenames.
        """
        return [os.path.basename(p) for p in self.paths]

    @property
    def ids(self):
        """
        Returns the IDs derived from filenames (filenames without extensions).

        Returns:
            list[str]: A list of IDs (filenames without extensions).
        """
        return [f.split('.')[0] for f in self.filenames]

    @property
    def target(self):
        """
        Gets the current target attribute.

        Returns:
            str: The current target variable (label column name, 'path', 'filename', 
            or 'id').
        """
        return self._target

    @target.setter
    def target(self, value: str):
        """
        Sets the target attribute and validates it.

        Args:
            value (str): The target label or attribute to be set.

        Raises:
            ValueError: If the provided target value is invalid.
        """
        valid = self.variables + [None, 'path', 'filename', 'id']
        if value not in valid:
            raise ValueError((f'Unable to set target {value}. '
                              f'Must be in {valid}'))

        self._target = value

    @property
    def y(self):
        """
        Returns the target variable based on the current target setting.

        Returns:
            np.ndarray or list[str]: The target variable (e.g., labels, paths, 
            filenames, ids). If `target` is `None`, returns an array of `None` values.
        """
        if self.target is None:
            return np.asarray([None] * len(self))
        elif self.target == 'path':
            return self.paths
        elif self.target == 'filename':
            return self.filenames
        elif self.target == 'id':
            return self.ids

        return self._labels[self.target]

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples (i.e., the number of image paths).
        """
        return len(self.paths)
