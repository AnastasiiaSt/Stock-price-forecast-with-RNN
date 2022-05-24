from tensorflow import keras
import mlflow


def build_model(
    x_train, y_train, n_hidden, n_neurons, learning_rate
):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(x_train.shape[1], x_train.shape[2])))
    for layer in range(n_hidden):
        model.add(keras.layers.LSTM(n_neurons))
    model.add(keras.layers.RepeatVector(y_train.shape[1]))
    model.add(keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=x_train.shape[2])))
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse")
    return model


def train_model(
    x_train,
    y_train,
    x_valid,
    y_valid,
    n_hidden,
    n_neurons,
    learning_rate,
    epochs,
    batch_size
):
    with mlflow.start_run():
        model = build_model(x_train, y_train, n_hidden, n_neurons, learning_rate,)
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid)
        )
        mlflow.tensorflow.autolog()


#model1 = build_model(learning_rate=0.00001)
#model1.set_weights(model.get_weights())
#model1.load_weights("Weights_3dims-05")

