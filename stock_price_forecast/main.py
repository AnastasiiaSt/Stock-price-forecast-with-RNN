import tensorflow as tf
import mlflow.tensorflow
from data import get_inputs, get_inputs_with_time
from preprocess import Preprocess
from train import train_model, train_cont


def main(
    p: int,
    t: int,
    valid_ratio: float,
    scaling: str,
    n_neurons: list,
    optimization: str,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    model_name: str,
    set_weights: bool,
    weights_name: str,
    inputs_name: str,
    fit_whole_dataset: bool,
):
    dataset = get_inputs_with_time("data", inputs_name)

    prep = Preprocess(p=p, t=t, valid_ratio=valid_ratio, scaling=scaling, split=True)
    dataset = dataset.drop(columns=["Open", "High", "Low"])
    prep.fit(dataset)
    (
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_train_indeces,
        x_valid_indeces,
        y_train_indeces,
        y_valid_indeces,
    ) = prep.transform(dataset)

    tf.random.set_seed(42)
    with mlflow.start_run(run_name = "train_valid_fit"):
        mlflow.log_param("train_timesteps", t)
        mlflow.log_param("predict_timesteps", p)
        mlflow.log_param("valid_percent", valid_ratio)
        mlflow.log_param("scaling", scaling)
        mlflow.log_param("n_neurons", n_neurons)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.tensorflow.autolog()

        model = train_model(
            x_train,
            y_train,
            x_valid,
            y_valid,
            n_neurons=n_neurons,
            optimization = optimization,
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            set_weights=set_weights,
            weights_name=weights_name,
        )

    model.save_weights("saved_data/weights_" + model_name)
    model.save("saved_data/model_" + model_name)

    if fit_whole_dataset:
        prep_without_split = Preprocess(
            p=p, t=t, valid_ratio=valid_ratio, scaling=scaling, split=False
        )
        prep_without_split.fit(dataset)
        (
            x_train,
            y_train,
            x_train_indeces,
            y_train_indeces,
        ) = prep_without_split.transform(dataset)

        tf.random.set_seed(42)
        with mlflow.start_run(run_name = "entire_dataset_fit"):
            mlflow.log_param("train_timesteps", t)
            mlflow.log_param("predict_timesteps", p)
            mlflow.log_param("scaling", scaling)
            mlflow.log_param("n_neurons", n_neurons)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.tensorflow.autolog()
            model = train_cont(
                x_train,
                y_train,
                n_neurons=n_neurons,
                optimization = optimization,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size,
                model_name=model_name,
            )

        model.save_weights("saved_data/entire_weights_" + model_name)
        model.save("saved_data/entire_model_" + model_name)


main(
    p=32,
    t=2048,
    valid_ratio=0.2,
    scaling="min_max",
    n_neurons=[32, 32],
    optimization = "Adam",
    learning_rate=0.0000001,
    epochs=20,
    batch_size=128,
    model_name="Coursera",
    set_weights=True,
    weights_name="Coursera",
    inputs_name="Coursera_15min.csv",
    fit_whole_dataset=False,
)
