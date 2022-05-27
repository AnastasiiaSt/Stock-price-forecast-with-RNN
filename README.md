### Description

The goal of this project is to create a deep learning model for Coursera stock price forecast. Dataset of stock prices with 15 minutes interval from 3 May 2021 to 1 January 2022 is used for training and validation. 
<img src="./images/historical prices.png" width="900">

Below presented monthly, weekly and daily graphs for the stock price just after Coursera IPO. Stock price graphs for other periods of time can be found in *visualization* package *Input Graphs* notebook.
<img src="./images/month graph.png" width="900">
<img src="./images/week graph.png" width="900">
<img src="./images/day graphs.png" width="900">

### Training

The dataset is divided into training and validation sets, which are used for model training. In the final step, entire dataset is fitted to the model. Long Short Term Memory (LSTM) model is implemented to make prediction of stock price for 32 consequent time steps, which are equivalent to 8 hours (32 timesteps times 15 minutes).

The training runs are stored in MLflow user interface, which is shown below.
<img src="./images/mlflow ui.png" width="900">

The final model show Mean Squared Error (MSE) of 0.0001 and Mean Absolute Error (MAE) of 0.008 on validation set.

### Prediction

Comparison of the model predictions with the actual stock price for the training and validation sets are depicted in the following diagrams. Predictions for other periods of time can be found in *visualization* package *Prediction Graphs* notebook.
<img src="./images/prediction train 1.png" width="900">
<img src="./images/prediction train 2.png" width="900">




