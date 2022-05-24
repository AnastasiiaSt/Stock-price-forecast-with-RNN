from data import get_inputs
from preprocess import Preprocess
from train import train_model


dataset = get_inputs("data", "BA daily.csv")
print(dataset)

prep = Preprocess()
prep.fit(dataset)
x_train, y_train, x_valid, y_valid = prep.transform(dataset)

train_model(
    x_train,
    y_train,
    x_valid,
    y_valid,
    n_hidden=1,
    n_neurons=32,
    learning_rate=0.0001,
    epochs = 100,
    batch_size = 128
)
