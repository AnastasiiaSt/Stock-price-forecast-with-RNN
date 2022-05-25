### Description

The goal of this project is to create a deep learning model for Boeing stock price forecast. The stock data are obtained from [Yahoo Finance](https://finance.yahoo.com/quote/BA/history?period1=473385600&period2=1653091200&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true). Stock prices from 1 January 1985 to 20 May 2022 are used for training and validation. 

<img src="./images/historical prices.png" width="900">

Long Short Term Memory (LSTM) model is implemented to make prediction of Close, High, Low and Open stock price for 5 consequent days. Hyperparameters such as number of hidden layers of the model, number of neurons, learning rate are tuned using Randomized Search cross-validation.

