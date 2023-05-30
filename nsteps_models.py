
from typing import Any

import torchmetrics as tm
import torch 

import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

from abstract_model import RegressionNet

class TimeSeriesFcRegressionN(RegressionNet): 
    def __init__(self, input_size, 
                 hidden_size, 
                 n_layers, 
                 n_steps,
                 loss=F.mse_loss):
        
        super().__init__(loss=loss)
        model = [nn.Linear(input_size, hidden_size)]
        model = [*model, *[nn.Linear(hidden_size, hidden_size) for layer in range(n_layers)]]
        model = [*model, *[nn.Linear(hidden_size, n_steps)]]
        
        self.regressor = nn.Sequential(*model)

    def forward(self, x): 
        return self.regressor(x.float())

    def configure_optimizers(self):
        return optim.Adam(self.regressor.parameters(), 
                          lr=3e-4)
    
    def validation_step(self, batch, batch_idx): 
        mse, mae = self._shared_eval_step(batch, batch_idx)
        metrics = {'mse': mse, 'mae': mae}
        self.log_dict(metrics,
                      prog_bar=True)
    
    def _shared_eval_step(self, batch, batch_idx): 
        x, y = batch 
        predictions = self(x)

        return tm.MeanSquaredError()(predictions, y),\
               tm.MeanAbsoluteError()(predictions, y)
    

class TimeSeriesLstmRegressionN(RegressionNet):
    def __init__(self, hidden_size): 
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.GRU() 
        self.decoder = nn.GRU() 
        self.regressor = nn.Linear(hidden_size)
    
    def forward(self, x): 
        encoder_h, hidden = self.encoder(x)
        decoder_h, _ = self.decoder(encoder_h, hidden)

        x_out = decoder_h.view(-1, self.hidden_size) 
        return self.regressor(x_out) 
