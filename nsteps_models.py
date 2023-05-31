
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
    def __init__(self, input_size,
                 hidden_size,
                #  decoder_hidden_size,
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
                          input_size=1)

        # self.decoder_hidden_size = decoder_hidden_size

        # self.enc_predictor = nn.Linear(self.scale_b*hidden_size, 1)

        # self.dec = nn.GRU(input_size=1, 
        #      hidden_size=decoder_hidden_size, 
        #      num_layers=1,
        #      batch_first=True,
        #      bidirectional=False)
         
        # self.regressor = nn.Sequential(
        #                 nn.Linear(decoder_hidden_size, decoder_hidden_size),
        #                 nn.LeakyReLU(),
        #                 nn.Linear(decoder_hidden_size, 1))
    
    def forward(self, x): 
        encoder_h, hidden = self.enc(x.float())
        
        # context_vector = hidden.mean(dim=0)
        # context_vector = context_vector.view(1,-1,self.decoder_hidden_size)
        # enc_preds = self.enc_predictor(encoder_h).view(-1, self.nsteps, 1)
        
        # decoder_h, _ = self.dec(enc_preds, context_vector)

        x_out = self.dec(encoder_h)
        return x_out.squeeze(2)
    
    def configure_optimizers(self) -> Any:
        return optim.Adam([*self.enc.parameters(), 
                           *self.dec.parameters(),
                        #    *self.enc_predictor.parameters(),
                        #    *self.regressor.parameters()
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