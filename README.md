# TIME SERIES FORECASTING

## <b>Approach:</b> 
```
This repository contains functionality for forecasting stock yield based on GRU (Gated Recurrent Unit) and CNN (Convolutional Neural Network).
The key idea is to build the encoder-decoder architecture of the RNN, as well as data-augmentation based on additional data sources.
```

## <b> Structure</b>: 
```
├── data [training source] 
├── fitted_models [weights of trained models] 
├── model_package:
├── ├── ...
├── ├── nsteps_models.py      [models for n-steps forecasting]
├── ├── single_step_models.py [models for one step forecasting]
├── ├── ...
├── data.py      [datasets for each model torch implementation]
├── utils.py     [visualization tools for timeseries forecasting]
└── launch.ipynb [usage example]
```




<a href="https://pytorchlightning.ai/index.html"><img src="https://tse3.mm.bing.net/th?id=OIP.KFKo1oaV1Pbrm3frVadQVAHaC8&pid=Api"></a>
<a href="https://pytorch.org/"><img src="https://d3njjcbhbojbot.cloudfront.net/api/utilities/v1/imageproxy/https://s3.amazonaws.com/coursera-course-photos/51/6d31a64dad46d08a076ef7abbf4f15/external-content.duckduckgo.com.jpg?auto=format%2Ccompress&dpr=2&w=150&h=150&fit=fill&bg=FFF&q=25i" height=169></a>
