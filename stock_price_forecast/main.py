from data import get_inputs
from preprocess import Preprocess
from train import train_model


def main():
    dataset = get_inputs("data", "BA daily.csv")
    print(dataset)

    prep = Preprocess(p=5, t=250, valid_ratio=0.10)
    prep.fit(dataset)
    x_train, y_train, x_valid, y_valid, x_train_indeces, x_valid_indeces, y_train_indeces, y_valid_indeces = prep.transform(dataset)

    model = train_model(
        x_train,
        y_train,
        x_valid,
        y_valid,
        n_neurons=[128,64,32,16],
        learning_rate=0.000002,
        epochs = 10000,
        batch_size = 128,
        set_weights = False
    )
 
    model.save_weights('saved_data/weights')
    model.save('saved_data/model')

main()
