import torch
from torch import nn
import torch.nn.functional as F
import torch.fft
import numpy as np
import torch.optim as optimizer
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import xarray as xr

from blocks import *
from NN import *


class Glonet(pl.LightningModule):
    def __init__(self, 
                 model_path : str, 
                 init_input : torch.Tensor):
        super().__init__()
        self.saved_model = torch.jit.load(model_path)
        
        # Freeze all parameters
        for param in self.saved_model.parameters():
            param.requires_grad = False
            
        # Add initial condition as a trainable parameter.
        self.init_input = nn.Parameter(init_input, requires_grad=True)
        
    def forward(self):
        # Use the optimizable initial condition instead of the input x
        # The idea is to optimize self.init_input to get better forecasts
        return self.saved_model(self.init_input)

    def step(self, 
             phase : str, 
             batch : torch.utils.data.DataLoader, 
             batch_idx : int) -> float :
        """Define commun step for training, validation and test phases."""
        
        x, y = batch
        y_hat = self.forward(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f"{phase}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss


    def training_step(self, batch, batch_idx) -> float :
        """Define training step."""
        
        phase = "train"
        return self.step(phase, batch, batch_idx)
    
    
    # def validation_step(self, batch, batch_idx) -> float :
    #     """Define validation step."""
        
    #     phase = "val"
    #     return self.step(phase, batch, batch_idx)


    def configure_optimizers(self) -> Dict[str, Any]:
        """Define optimizers and learning rate schedulers."""

        optimizer = torch.optim.Adam([self.init_input], lr=1e-4, weight_decay=1e-5, betas=(0.9, 0.999))
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    
class OptimizeInitialConditionDataset(torch.utils.data.Dataset):
    
    def __init__(self,
                 data_path : str,
                 input_file : str = "input1.nc",  # Which input file to use
                 sample_index : int = 0,  # Which sample to optimize
                 variable : str = "data", 
                 time_dim : str = "time",
                 space_dim : Tuple[str] = ("latitude", "longitude"), 
                 forecast_horizon : int = 7, 
                 standardization : bool = True,
                 random_seed : int = 42) :

        self.data_path = data_path
        self.input_file = input_file
        self.sample_index = sample_index
        self.variable = variable
        self.time_dim = time_dim
        self.space_dim = space_dim
        self.forecast_horizon = forecast_horizon
        self.standardization = standardization
        self.random_seed = random_seed
        
        self.sequence_length = 2  # Length of the input sequence

        self.data = self._load_data()
        self.data = self._preprocess_data()
        self._make_valid_indices()
        
        # Get the specific sample
        self.input_sequence, self.target = self._get_sample()
        

    def _load_data(self) -> xr.Dataset :
        try :
            dataset = xr.open_dataset(f"{self.data_path}/{self.input_file}")
            print(f"Loaded {self.input_file} successfully")
        except FileNotFoundError as e:
            print(f"Error loading dataset {self.input_file}: {e}")
            raise
        
        return dataset

    def _preprocess_data(self) -> torch.Tensor :
        """Preprocess the loaded data"""
        
        data = self.data["data"]
        # Convert to torch tensor
        data = torch.from_numpy(data.values).float()
        
        return data
        
    def _make_valid_indices(self) :

        T = self.data.shape[0]
        min_length = self.sequence_length + self.forecast_horizon
        self.valid_indices = list(range(T - min_length + 1))

    def _get_sample(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the specific sample to optimize."""
        
        if self.sample_index >= len(self.valid_indices):
            raise ValueError(f"Sample index {self.sample_index} out of range. Max: {len(self.valid_indices)-1}")
        
        start_idx = self.valid_indices[self.sample_index]
        input_sequence = self.data[start_idx:start_idx + self.sequence_length]
        
        target_idx = start_idx + self.sequence_length + self.forecast_horizon - 1
        target = self.data[target_idx]
        
        return input_sequence, target
            
    def __len__(self) -> int :
        return 1  # Only one sample
    
    def __getitem__(self, idx : int) -> Tuple[torch.Tensor, torch.Tensor] :
        # Always return the same sample
        return self.input_sequence, self.target


class GlorysDataModule(pl.LightningDataModule) :
    def __init__(self, data_dir, input_file="input1.nc", sample_index=0, batch_size=1, num_workers=0) :
        super().__init__()
        self.data_dir = data_dir
        self.input_file = input_file
        self.sample_index = sample_index
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None) :
        self.train_dataset = OptimizeInitialConditionDataset(
            self.data_dir, 
            input_file=self.input_file,
            sample_index=self.sample_index
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return None  # No validation for single sample optimization 