
import pandas as pd

import torch 
from torch.utils.data import Dataset 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from datetime import time


class TimeSeriesDataset(Dataset): 
    
    def __init__(self, paths, augm_path, 
                 window_len:int) -> None: 
        self.window_len = window_len
        data = self.prepare(paths)
        self.augmented_data = self.augment_data(augm_path)

        data = data.merge(self.augmented_data,
                          left_index=True,
                          right_on='DATE', 
                          how='left').dropna()

        y, scaler = self.scale_data_to_train(data['D'])

        X = [] 
        for col in data: 
            if col != 'DATE':
                X_tmp = self.to_autoregression_form(pd.DataFrame(data[col]),
                                                window_len, 
                                                target_column=col
                                                ).dropna().reset_index(drop=True)
                X.append(X_tmp)
            

        self.X = pd.concat(X, axis=1)
        x_scaler = StandardScaler() 
        self.X = pd.DataFrame(x_scaler.fit_transform(self.X),
                              columns=self.X.columns)

        self.data = pd.concat([self.X, y], axis=1).dropna()
        self.X, self.y = self.data.drop('D', axis=1), self.data['D']
        

    def __getitem__(self, index):
        return torch.from_numpy(self.X.iloc[index].values).view(-1,self.window_len).transpose(1, 0), \
               torch.tensor([self.y.iloc[index]])
    
    def __len__(self): 
        return len(self.data)
    
    @staticmethod
    def to_autoregression_form(data:pd.DataFrame,
                               window_len:int,
                               target_column:str='D') -> pd.DataFrame:
        X = pd.concat([pd.DataFrame(data[target_column].shift(w)).rename({target_column: f"{target_column}_lag{w}"}, axis=1) \
                          for w in range(window_len, 0, -1)],
                          axis=1) 
        return X
    
    @staticmethod
    def prepare(paths) -> pd.DataFrame: 
        data = pd.concat(map(lambda pth: pd.read_csv(pth, sep=';',
                                               parse_dates=['DATE'],
                                               dayfirst=True), paths)).reset_index(drop=True)
        
        data['TIME'] = data['TIME'].map(lambda date_string: time(*map(int, date_string.split(":"))))
        data = data.sort_values(by=['DATE', 'TIME'])
        data.set_index(keys=['DATE', 'TIME'], inplace=True)
        data = data.groupby(level=0).agg(lambda obs: obs[-1])
        data['D'] = data['CLOSE'].pct_change()
        # data['D'] = data.D.shift(-1)
        data.dropna(inplace=True)
        data = data[['D']]
        return data
    
    @staticmethod
    def scale_data_to_train(y):
        q3 = np.quantile(y, 0.75)
        q1 = np.quantile(y, 0.25)
        iqr = q3 - q1
        outlayer_th = q1 - 1.5*iqr, q3 + 1.5*iqr

        y_mask_more = y > outlayer_th[1]  
        y_mask_less = y < outlayer_th[0]

        y[y_mask_more | y_mask_less] = y[~y_mask_more & ~y_mask_less].median()

        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.values.reshape(-1,1))
        # y_scaled = np.exp(y_scaled)
        y = pd.Series(y_scaled.reshape(-1), name=y.name)
        # y = np.log(2+y)

        return y, scaler
    
    @staticmethod
    def augment_data(path: str): 
        excel_file = pd.ExcelFile(path)
        for num, sheet in enumerate(excel_file.sheet_names): 
            if num == 0: 
                df_initial = pd.read_excel(excel_file, sheet,
                                parse_dates=['DATE']).rename({'VALUE': sheet},
                                                                axis=1)
            else: 
                df_tmp = pd.read_excel(excel_file, sheet,
                                    parse_dates=['DATE']).rename({'VALUE': sheet},
                                                                    axis=1)
                df_initial = df_initial.merge(df_tmp, how='left',
                                            left_on='DATE', 
                                            right_on='DATE')
        
        df_filtered = df_initial.loc[:,df_initial.columns[~(df_initial.isna().sum()/len(df_initial) > 0.5)]]
        na_columns = df_filtered.columns[df_filtered.isna().sum() > 0]
        for na_col in na_columns: 
            df_filtered.loc[df_filtered[na_col].isna(),na_col] = df_filtered[na_col].median()
        
        return df_filtered
    

class TimeSeriesConv1dDataset(TimeSeriesDataset):
    def __init__(self, paths, augm_path, 
                 window_len:int) -> None: 
        super().__init__(paths, augm_path, window_len) 

    def __getitem__(self, index:int): 
        return torch.from_numpy(self.X.iloc[index].values), \
               torch.tensor([self.y.iloc[index]])
    

class TimeSeriesNstepsDataset(TimeSeriesDataset): 

    def __init__(self, paths, augm_path, window_len):
        super().__init__(paths, augm_path, window_len)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.X.iloc[index].values), \
               torch.tensor(self.y.values[index:(index+self.window_len)])
    
    def __len__(self): 
        return len(self.data) - self.window_len
    

class TimeSeriesNstepsLstmDataset(TimeSeriesDataset): 
    
    def __init__(self, paths, augm_path, window_len):
        super().__init__(paths, augm_path, window_len)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.X.iloc[index].values).view(-1, self.window_len).transpose(1,0), \
               torch.tensor(self.y.values[index:(index+self.window_len)])
    
    def __len__(self): 
        return len(self.data) - self.window_len
    
    
class SberDataset: 

    def __init__(self, sber_path, window_len) -> None: 
        self.sber_path = sber_path
        self.window_len = window_len
        self.prepare(sber_path)
        self.X = TimeSeriesDataset.to_autoregression_form(data=self.data, target_column='Close',
                                                             window_len=self.window_len)
        self.data = pd.concat([self.X, self.data['Close']], axis=1).rename({"Close": "y"}, axis=1).dropna() 
        self.y = self.data['y']
        self.X = self.data.drop('y', axis=1)

    def __getitem__(self, index):
        return torch.from_numpy(self.X.iloc[index].values.reshape(-1,1)), \
               torch.Tensor(self.y.iloc[index: (index+self.window_len)])
    
    def __len__(self): 
        return len(self.X) - self.window_len 


    def prepare(self, path): 
        sber_path = sorted(path.glob('*.txt'))
        data = pd.read_csv(sber_path[0], sep='\t', header=None,
                        parse_dates=[0], 
                        names=['Datetime', 'Open', 'High', 'Low', 'Close'])
        data = data[['Datetime', 'Close']]
        data['Date'] = data['Datetime'].map(lambda dt: dt.date())
        data = data.sort_values(by=['Datetime'])
        data.drop("Datetime", axis=1, inplace=True)
        data.set_index("Date", inplace=True)

        data = data.groupby(level=0).agg({"Close": lambda rows: rows[-1]})
        data['Close'] = data['Close'].pct_change()
        data.dropna(inplace=True)

        self.data = data 
