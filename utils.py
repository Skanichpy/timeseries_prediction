import torch 
import matplotlib.pyplot as plt 

import numpy as np

def time_series_split(dataset, test_size: float): 
    test_count = int(len(dataset) * test_size)
    train_count = len(dataset) - test_count

    return torch.utils.data.random_split(dataset, lengths=[train_count, test_count])


def get_gt(dataset, mode='one_step'):
    if mode == 'one_step':
        return torch.Tensor([p[1] for p in dataset])
    elif mode == 'n_step': 
        return torch.cat([p[1].view(1,-1) for p in dataset], 
                         dim=0)
    else: 
        raise ValueError('mode must be in {one_step, n_step}')

def plot_real_and_predictions(test_dataset, train_dataset,
                              train_predictions, test_predictions):
    
    def inverse(x):
        return x

    train_gt, test_gt = get_gt(train_dataset), get_gt(test_dataset)
    train_predictions = torch.cat(train_predictions) 
    test_predictions = torch.cat(test_predictions)

    x_plot_train = range(len(train_predictions))
    x_plot_test = range(len(train_predictions), len(train_predictions) + len(test_predictions))
    
    train_gt = inverse(train_gt.cpu().detach().numpy()) 
    test_gt = inverse(test_gt.cpu().detach().numpy()) 
    
    test_predictions = inverse(test_predictions.cpu().detach().numpy()) 
    train_predictions = inverse(train_predictions.cpu().detach().numpy()) 

    plt.title('GT AND ESTIMATION OF TIME SERIES')
    plt.fill_between(x_plot_train, 
                     y1=min(train_gt), y2=max(train_gt), 
                     alpha=.2, 
                     label='TRAIN SPACE')
    plt.plot(x_plot_train, 
             train_gt, 
             label='gt',
             linewidth=12)
    
    plt.plot(x_plot_train, 
             train_predictions, 
             label='estimation')

    plt.fill_between(x_plot_test, 
                     y1=min(train_gt), y2=max(train_gt), 
                     alpha=.2, 
                     label='TEST SPACE')
    plt.plot(x_plot_test, 
             test_gt, 
             label='gt',
             linewidth=12)
    plt.plot(x_plot_test, 
             test_predictions, 
             label='estimation')
    
    plt.legend()

def plot_nstep_real_and_predictions(test_dataset, test_predictions):
    fig, ax = plt.subplots(nrows=len(test_dataset)-1, ncols=1,
                           figsize=(12, 100))
    test_gt = get_gt(test_dataset, mode='n_step')
    for idx in range(1, len(test_gt)): 
        tmp_before = test_gt[idx-1]
        tmp_after = test_gt[idx]

        x_before = range(len(tmp_before))
        x_after = range(len(x_before)-1, len(x_before)+len(tmp_after))

        
        ax[idx-1].plot(x_before, tmp_before, label='train gt')
        ax[idx-1].plot(x_after,
                torch.cat([tmp_before[-1].view(-1,1), 
                        tmp_after.view(-1,1)]),
                linewidth=4,
                label='test gt')
        ax[idx-1].plot(x_after[1:], test_predictions[idx],
                       label='predictions')
        ax[idx-1].set_title(f'GT and PRED for step: {idx}') 
        ax[idx-1].legend() 
