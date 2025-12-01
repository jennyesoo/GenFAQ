from datasets import load_dataset
from .formatting import chatml_format


def load_and_format(dataset_name: str, split: str):
    ds = load_dataset(dataset_name, split=split)
    original_cols = ds.column_names
    
    ds = ds.map(function=chatml_format, remove_columns=original_cols)
    return ds