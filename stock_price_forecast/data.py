import pandas as pd
import os
from pathlib import Path


def get_inputs(file_path: str, file_name: str):
    full_path = os.path.join(Path.cwd(), file_path, file_name)
    dataset = pd.read_csv(full_path)
    dataset.set_index(pd.to_datetime(dataset["Date"], format="%m/%d/%Y"), inplace=True)
    dataset.drop(columns=["Date", "Adj Close", "Volume"], inplace=True)
    return dataset
