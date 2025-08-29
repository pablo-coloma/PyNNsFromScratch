from mynns import MLP
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import time

# Dataset:
# https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz

def load_mnist():
    path = "examples/data/mnist.npz"
    with np.load(path) as f:
        X_train, y_train = f["x_train"], f["y_train"]
        X_test, y_test = f["x_test"], f["y_test"]
    return (X_train, y_train), (X_test, y_test)


def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


def test_my_net(X_train, y_train, X_test, y_test, plot=False, verbose=False):

    print("\n##### My MLP Classifier #####\n")

    mynet = MLP(
        layer_sizes=[28 * 28, 256, 128, 10],
        activation_function="sigmoid",
        task="multiclass",
        optimization_method="Adam",
        learning_rate=1e-2,
        l2_lambda=1e-5,
        seed=42
    )

    time0 = time.time()
    mynet.fit(X_train, y_train, epochs=10, batch_size=2000, verbose=verbose)
    time1 = time.time()
    print(f"Time to train my net: {time1-time0:.2f}")
    # About 30s

    preds = mynet.predict(X_test)
    acc = np.mean(np.argmax(preds, axis=1) == y_test)
    print(f"Test accuracy on my net: {acc:.4f}\n")

    if plot:
        mynet.plot_network(title="NN after train",
                    edge_threshold=0.05,
                    node_size=50,
                    max_edge_width=1.0)
    
    return mynet


def test_sck_net(X_train, y_train, X_test, y_test):

    print("\n##### Scklearn MLP Classifier #####\n")

    scknet = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="logistic",
        solver="adam",
        learning_rate_init=1e-2,
        batch_size=2000,
        max_iter=200,
        early_stopping=True,
        shuffle=True,
        random_state=42,
        beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )
    time0 = time.time()
    scknet.fit(X_train, y_train)
    time1 = time.time()
    print(f"Time to train scklearn net: {time1-time0:.2f}")

    preds = scknet.predict(X_test)
    acc = np.mean(preds == y_test)
    print(f"Test accuracy on scklearn net: {acc:.4f}\n")

    return scknet


def plot_predictions(net, X_test, y_test, seed=42):

    if isinstance(net, MLP):
        preds = net.predict(X_test)
        preds = np.argmax(preds, axis=1)
        title = "My MLP Classifier Predictions 0-9"
    elif isinstance(net, MLPClassifier):
        preds = net.predict(X_test)
        title = "Scklearn MLP Classifier Predictions 0-9"
    else:
        return

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 10, figsize=(16, 4))

    for digit in range(10):
        # Correct examples
        correct_idx = np.where((y_test == digit) & (preds == digit))[0]
        if len(correct_idx) > 0:
            idx = rng.choice(correct_idx)
            ax = axes[0, digit]
            ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"Prediction:{preds[idx]}", color="green")
            ax.axis("off")
        else:
            axes[0, digit].axis("off")

        # Incorrect examples
        wrong_idx = np.where((y_test == digit) & (preds != digit))[0]
        if len(wrong_idx) > 0:
            idx = rng.choice(wrong_idx)
            ax = axes[1, digit]
            ax.imshow(X_test[idx].reshape(28, 28), cmap="gray")
            ax.set_title(f"Prediction:{preds[idx]}", color="red")
            ax.axis("off")
        else:
            axes[1, digit].axis("off")

    fig.suptitle(title, fontsize=16)
    fig.text(0.04, 0.63, "Correct", va="center", ha="center", 
            fontsize=14, color="green", rotation=90)
    fig.text(0.04, 0.27, "Wrong", va="center", ha="center", 
            fontsize=14, color="red", rotation=90)

    plt.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])


def main():

    (X_train, y_train), (X_test, y_test) = load_mnist()

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    y_train_oh = one_hot(y_train, 10)

    mynet = test_my_net(X_train, y_train_oh, X_test, y_test, plot=False, verbose=False)
    # Time 30s, Accuracy 97.7 %
    scknet = test_sck_net(X_train, y_train, X_test, y_test)
    # Time 45s, Accuracy 97.9 %

    plot_predictions(mynet, X_test, y_test)
    plot_predictions(scknet, X_test, y_test)
    plt.show()


if __name__ == "__main__":
    main()