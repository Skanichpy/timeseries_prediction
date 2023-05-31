import torchmetrics as tm 
import torch 

import lightning as pl



class RegressionNet(pl.LightningModule): 
    
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def training_step(self, batch, batch_idx): 
            X_batch, y_batch = batch 
            model_outputs = self.forward(X_batch)

            loss = self.loss(model_outputs, y_batch.float())
            self.log('train_l2_loss', loss, on_step=True, on_epoch=True,
                    prog_bar=True, logger=False)
            return loss

    def validation_step(self, batch, batch_idx): 
        r2score, mse, mae = self._shared_eval_step(batch, batch_idx)
        metrics = {'r2_score': r2score, 'mse': mse, 'mae': mae}
        self.log_dict(metrics,
                      prog_bar=True)

    def _shared_eval_step(self, batch, batch_idx): 
        x, y = batch 
        predictions = self(x)

        return tm.R2Score()(predictions, y),\
               tm.MeanSquaredError()(predictions, y),\
               tm.MeanAbsoluteError()(predictions, y)
    
    def predict_step(self, batch, batch_idx:int): 
        x, _ = batch
        return self(x)