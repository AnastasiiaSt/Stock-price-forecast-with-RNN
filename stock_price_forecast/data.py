import pandas as pd
import os
from pathlib import Path


def get_inputs(file_name: str):
    full_path = os.path.join(Path.cwd(), "data", file_name)
    dataset = pd.read_csv(full_path)
    dataset.set_index(pd.to_datetime(dataset["Date"], format="%m/%d/%Y"), inplace=True)
    dataset.drop(columns=["Date", "Adj Close", "Volume"], inplace=True)
    return dataset


def get_inputs_with_time(file_path: str, file_name: str) -> pd.DataFrame:
    full_path = os.path.join(Path.cwd(), file_path, file_name)
    dataset = pd.read_csv(full_path)
    datetime_list = []
    for i in range(len(dataset["Date"])):
        day, month, year = dataset["Date"][i].split("/")
        if len(day) == 1:
            day = str(0) + day
        if len(month) == 1:
            month = str(0) + month
        if len(year) == 2:
            year = str(20) + year
        datetime_list.append(month + "/" + day + "/" + year + " " + dataset["Time"][i])
    dataset["DateTime"] = datetime_list
    dataset.set_index(
        pd.to_datetime(dataset["DateTime"], format="%m/%d/%Y %H:%M:%S"), inplace=True
    )
    dataset.drop(columns=["Date", "Time", "DateTime", "Volume"], inplace=True)

    return dataset
