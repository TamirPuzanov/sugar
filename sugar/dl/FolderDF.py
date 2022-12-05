import pandas as pd
from pathlib import Path

from torch.utils.data import Dataset
import torch
import os


class FolderDF(Dataset):
    def __init__(self, root_path: str, df: pd.DataFrame, csv_path: str = None, 
        path_column: str = "path", label_column:str = "label",
        open_fn=None, file_formats: list[str] = ["png", "jpg", "jpeg"]
    ) -> None:
        super().__init__()

        assert open_fn is not None

        if csv_path is not None:
            self.df = pd.read_csv(csv_path)[[path_column, label_column]]
        else:
            self.df = df[[path_column, label_column]]

        self.root_path = root_path
        self.classes = list(self.df[label_column].unique())

        self.path_column = path_column
        self.label_column = label_column

        self.open_fn = open_fn
        self.df = self.df.to_numpy()

        self.file_formats = file_formats

    def __len__(self):
        return len(self.df)
    
    def forward(self, idx):
        try:
            file_path = os.path.join(self.root_path, self.df[idx][0])
            file_ = self.open_fn(file_path)

            if self.file_formats is not None:
                if file_path.split(".")[-1] not in self.file_formats:
                    raise

            return file_, self.classes.index(self.df[idx][1])
        except:
            return self.forward((idx + 1) % self.__len__())
