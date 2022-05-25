import tensorflow as tf
import mlflow
import mlflow.tensorflow


def build_model(
    x_train, y_train, n_hidden, n_neurons, learning_rate
):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], x_train.shape[2])))
    for layer in range(n_hidden):
        model.add(tf.keras.layers.LSTM(n_neurons))
    model.add(tf.keras.layers.RepeatVector(y_train.shape[1]))
    model.add(tf.keras.layers.LSTM(n_neurons, return_sequences=True))
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=x_train.shape[2])))
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss="mse", optimizer = optimizer)
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
    batch_size,
    set_weights
):
    tf.random.set_seed(42)
    with mlflow.start_run():
        mlflow.tensorflow.autolog()
        model = build_model(x_train, y_train, n_hidden, n_neurons, learning_rate)
        if set_weights:
            model.load_weights('saved_data/weights')
        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_valid, y_valid)
        )
    return model


#model1 = build_model(learning_rate=0.00001)
#model1.set_weights(model.get_weights())
#model1.load_weights("Weights_3dims-05")

