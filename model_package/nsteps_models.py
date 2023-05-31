
from typing import Any

import torchmetrics as tm
import torch 

import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 

from . import abstract_model 

class TimeSeriesFcRegressionN(abstract_model.RegressionNet): 
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
    

class TimeSeriesGruRegressionN(abstract_model.RegressionNet):
    def __init__(self, input_size,
                 hidden_size,
                 num_layers,
                 bidirectional,
                 nsteps,
                 loss=F.mse_loss): 
        super().__init__(loss)
        self.hidden_size = hidden_size

        self.enc = nn.GRU(input_size=input_size,
             hidden_size=hidden_size,
             num_layers=num_layers,
             batch_first=True,
             bidirectional=bidirectional) 
        
        

        self.nsteps = nsteps
        self.scale_b = 2 if bidirectional else 1 
        self.new_input_size = self.scale_b * self.hidden_size
        self.fc_size = self.scale_b * hidden_size

        self.dec = nn.GRU(batch_first=True,
                          hidden_size=hidden_size,
                          input_size=hidden_size, 
                          num_layers=1)
        
        self.additional_regressor = nn.Sequential(nn.Linear(self.nsteps, self.nsteps))
        
        self.regressor = nn.Sequential(
                            nn.Linear(self.hidden_size, 1))
    
    def forward(self, x): 
        encoder_h, context = self.enc(x.float())
        encoder_h = encoder_h[:,-1,:].view(-1,1,self.hidden_size)
        
        outputs = []
        for step in range(self.nsteps):
            encoder_h, context = self.dec(encoder_h[:,-1,:].view(-1, 1, self.hidden_size), context)
            outputs.append(encoder_h.view(-1, self.hidden_size))
        
        x_out = torch.cat(outputs).view(-1, self.nsteps, self.hidden_size)
        out = self.regressor(x_out).squeeze(2)
        out2 = self.additional_regressor(x.squeeze(2).float())
        return out/2 + out2/2
    
    def configure_optimizers(self) -> Any:
        return optim.Adam([*self.enc.parameters(), 
                           *self.dec.parameters(),
                           *self.additional_regressor.parameters(),
                           *self.regressor.parameters()
                          ],
                           lr=0.1)


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