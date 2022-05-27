import tensorflow as tf
import mlflow
import mlflow.tensorflow


def build_model(x_train, y_train, n_neurons, optimization, learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=(x_train.shape[1], 1)))
    for i in range(len(n_neurons) - 1):
        model.add(tf.keras.layers.LSTM(n_neurons[i], return_sequences=True))
    model.add(tf.keras.layers.LSTM(n_neurons[-1]))
    model.add(tf.keras.layers.Dense(units=y_train.shape[1]))

    if optimization == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    elif optimization == "SGD":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=50, decay_rate=0.95
        )
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)
    else:
        raise ValueError("Defined optimization option is not available" % (optimization))

    model.compile(loss="mse", metrics = ["mae", "mape"], optimizer=optimizer)
    return model

def train_model(
    x_train,
    y_train,
    x_valid,
    y_valid,
    n_neurons,
    optimization,
    learning_rate,
    epochs,
    batch_size,
    set_weights,
    weights_name,
):
    model = build_model(x_train = x_train, y_train = y_train, n_neurons = n_neurons, optimization = optimization, learning_rate = learning_rate)
    if set_weights:
        model.load_weights("saved_data/weights_" + weights_name)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_valid, y_valid),
    )
    return model


def train_cont(
    x_train, y_train, n_neurons, optimization, learning_rate, epochs, batch_size, model_name
):

    model = build_model(x_train = x_train, y_train = y_train, n_neurons = n_neurons, optimization = optimization, learning_rate = learning_rate)
    model.load_weights("saved_data/weights_" + model_name)
    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
    )
    return model
