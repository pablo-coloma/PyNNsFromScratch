from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from mynns import MLP
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import time

# Dataset:
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/california_housing.npz


def load_ch():
    path = "examples/data/california_housing.npz"
    with np.load(path) as f:
        X, y = f["x"], f["y"].reshape(-1, 1)
    return X, y


def main():

    seed = 42

    X, y = load_ch()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train = X_scaler.transform(X_train)
    X_test  = X_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train)

    model = MLP(
        layer_sizes=[8, 128, 64, 1],
        activation_function="relu",
        task="regression",
        optimization_method="Adam",
        learning_rate=1e-3,
        seed=seed
    )

    t0 = time.time()
    model.fit(X_train, y_train_s, batch_size=256, epochs=300)
    t1 = time.time()
    print(f"Time to train: {t1 - t0:.2f}s")

    y_pred = y_scaler.inverse_transform(model.predict(X_test))

    mae  = mean_absolute_error(y_test, y_pred)
    print(f"MAE: {mae:.4f}")

    residuals = y_pred - y_test

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("California Housing - Performance on Test", fontsize=16)

    formatter = FuncFormatter(lambda x, _: f"{int(x/1000)}k$")

    axes[0].scatter(y_test, y_pred, alpha=0.5, s=12)
    min_v = min(y_test.min(), y_pred.min())
    max_v = max(y_test.max(), y_pred.max())
    axes[0].plot([min_v, max_v], [min_v, max_v], linewidth=2, color="red")
    axes[0].set_xlabel("Real value")
    axes[0].set_ylabel("Prediction")
    axes[0].set_title("Prediction vs. Real Value")
    axes[0].xaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(formatter)

    axes[1].hist(residuals, bins=40, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Error (pred - real)")
    axes[1].set_ylabel("Frecuency")
    axes[1].set_title("Residuals")
    axes[1].xaxis.set_major_formatter(formatter)

    plt.tight_layout()
    plt.show()

    # Time to train: 31.39s
    # MAE: 36179.4080


if __name__ == "__main__":
    main()