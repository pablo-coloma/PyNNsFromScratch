from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from mynns import MLP
import time


def main():
    seed = 42

    data = load_breast_cancer()
    X, y = data.data, data.target.reshape(-1, 1) # (569, 30), (569,  1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)

    mynet = MLP(
        layer_sizes=[30, 64, 32, 1],
        activation_function="relu",
        task="multilabel",
        optimization_method="Adam",
        learning_rate=1e-2,
        seed=seed
    )

    time0 = time.time()
    mynet.fit(X_train, y_train, batch_size=64, epochs=200)
    time1 = time.time()
    print(f"Time to train the net: {time1-time0:.2f}")

    prob = mynet.predict(X_test).reshape(-1)
    pred  = (prob >= 0.5).astype(int)

    acc = (pred == y_test.ravel()).mean()
    auc = roc_auc_score(y_test, prob)
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {auc:.4f}")

    cm  = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Malignant", "Benign"],
                yticklabels=["Malignant", "Benign"])
    plt.xlabel("Pred")
    plt.ylabel("Real")
    plt.title("Breast Cancer Classifier Test")
    plt.show()

    # Time 0.5s, Accuracy 97.2 %


if __name__ == "__main__":
    main()