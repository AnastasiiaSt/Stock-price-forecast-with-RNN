### Description

The goal of this project is to create a deep learning model for Coursera stock price forecast. Dataset of stock prices with 15 minutes interval from 3 May 2021 to 1 January 2022 is used for training and validation. 

<img src="./images/historical prices.png" width="900">

Time step is equal to 15 minutes. Long Short Term Memory (LSTM) model is implemented to make prediction of stock price for 32 consequent time steps, which are equivalent to 8 hours (32 timesteps times 15 minutes).

