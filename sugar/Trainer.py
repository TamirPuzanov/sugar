import torch.nn as nn
import torch

from .log import TQDM, Log_Interface
from .nn import Module


class Trainer:
    def __init__(self, model: Module, optim: torch.optim.Optimizer = None, criterion = nn.MSELoss,
        device = torch.device("cpu"), log_interface: Log_Interface = TQDM) -> None:

        self.model = model
        self.criterion = criterion

        self.log_interface = log_interface()
        self.device = device

        if optim is None:
            self.optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optim = optim

    def fit(self, num_epochs, train_dl: torch.utils.data.DataLoader, 
        valid_dl, log_interval = 5, metrics: list[str] = []):

        for epoch in range(num_epochs):
            self.log_interface.start_batch_iteration()
            self.train_batch(train_dl)
    
    def train_batch(self, train_dl):
        for i, batch in enumerate(train_dl):
            if len(batch.shape) == 2:
                X = batch[0].to(self.device)
                y = batch[0].to(self.device)