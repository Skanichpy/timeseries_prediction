import lightning as pl
import torch

import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim 

import torchmetrics as tm

from abstract_model import RegressionNet


    

class TimeSeriesLstmRegression(RegressionNet):
    def __init__(self, input_size:int, hidden_size:int,
                 out_layer_size:int, num_layers:int,
                 bidirectional:bool, loss=F.mse_loss): 
        super().__init__(loss)
        self.features_extractor = nn.GRU(input_size=input_size,
                                          hidden_size=hidden_size,
                                          batch_first=True,
                                          num_layers=num_layers,
                                          bidirectional=bidirectional)
        
        self.hidden_size = hidden_size 
        self.bidir_scale= 2 if bidirectional else 1
        self.num_rnn_layers = num_layers
        self.regressor = nn.Sequential(

            nn.Dropout(p=.3),
            nn.Linear(hidden_size*num_layers*self.bidir_scale, out_layer_size),
            nn.LeakyReLU(inplace=True),

            nn.Linear(out_layer_size, 1)
        )

    def forward(self, x): 
        _, hidden = self.features_extractor(x.float())
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.reshape(-1, self.bidir_scale * self.num_rnn_layers * self.hidden_size)
        
        return self.regressor(hidden) #.squeeze(0)

    def configure_optimizers(self):
        return optim.Adam([*self.features_extractor.parameters(), *self.regressor.parameters()],
                          lr=3e-4)


class TimeSeriesConv1dRegression(RegressionNet): 

    def __init__(self, in_channels, out_features, loss=F.mse_loss): 
        super().__init__(loss)
            
        self.features_extractor = nn.Sequential(nn.Conv1d(in_channels=in_channels, 
                                                          out_channels=64,
                                                          kernel_size=2,
                                                          stride=1,
                                                          padding=0),
                                                nn.BatchNorm1d(64),
                                                nn.AvgPool1d(2),
                                                nn.ReLU(),

                                                nn.Conv1d(in_channels=64, out_channels=128,
                                                          stride=1, padding=0,
                                                          kernel_size=2),
                                                nn.BatchNorm1d(128),
                                                nn.AvgPool1d(2),
                                                ) 

        self.regressor = nn.Linear(3456, 1)


    def forward(self, x): 
        x = x.unsqueeze(1).float()
        features = self.features_extractor(x)
        features = nn.Flatten()(features)

        outputs = self.regressor(features)
        return outputs

    def configure_optimizers(self): 
        return optim.Adam([*self.features_extractor.parameters(), *self.regressor.parameters()],
                          lr=0.01)