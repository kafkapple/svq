# Empty init file to make data a proper Python package

from .datasets import get_dataset, get_data_loaders, ShapesDataset, MultiDspritesDataset

__all__ = [
    'get_dataset',
    'get_data_loaders',
    'ShapesDataset',
    'MultiDspritesDataset'
]
