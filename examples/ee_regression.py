from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from mynns import MLP
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Link:
# https://archive.ics.uci.edu/ml/datasets/energy+efficiency


def main():

    df = pd.read_excel("examples/data/ENB2012_data.xlsx")

    X = df.iloc[:, :-2].to_numpy() # features
    y = df.iloc[:, -2:].to_numpy() # heating & cooling loads

    seed = 42

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)

    X_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    X_train = X_scaler.transform(X_train)
    X_test  = X_scaler.transform(X_test)
    y_train_s = y_scaler.transform(y_train)

    model = MLP(
        layer_sizes=[8, 128, 64, 2],
        activation_function="relu",
        task="regression",
        optimization_method="Adam",
        learning_rate=1e-3,
        seed=seed
    )

    t0 = time.time()
    model.fit(X_train, y_train_s, batch_size=64, epochs=800)
    t1 = time.time()
    print(f"Time to train: {t1 - t0:.2f}s")

    y_pred_s = model.predict(X_test)
    y_pred   = y_scaler.inverse_transform(y_pred_s)

    mae_each = mean_absolute_error(y_test, y_pred, multioutput="raw_values")
    print(f"MAE Heating: {mae_each[0]:.4f}")
    print(f"MAE Cooling: {mae_each[1]:.4f}")

    residuals = y_pred - y_test
    names = ["Heating Load", "Cooling Load"]

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    fig.suptitle("Energy Efficiency — Performance on Test", fontsize=16)

    for i, name in enumerate(names):
        ax = axes[0, i]
        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=12)
        min_v = min(y_test[:, i].min(), y_pred[:, i].min())
        max_v = max(y_test[:, i].max(), y_pred[:, i].max())
        ax.plot([min_v, max_v], [min_v, max_v], linewidth=2, color="red")
        ax.set_xlabel(f"Real value")
        ax.set_ylabel("Prediction")
        ax.set_title(f"Prediction vs. Real — {name}")
        ax.grid(True, linestyle="--", alpha=0.3)

    for i, name in enumerate(names):
        ax = axes[1, i]
        ax.hist(residuals[:, i], bins=40, edgecolor="black")
        ax.set_xlabel(f"Error (pred - real)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Residuals — {name}")
        ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Time: 4s
    # MAE Heating: 0.6241
    # MAE Cooling: 1.0415


if __name__ == "__main__":
    main()