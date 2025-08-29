import numpy as np
import pickle
from typing import Union, Tuple, Optional
from .functions import (
    sigmoid, d_sigmoid, relu, d_relu, tanh, d_tanh,
    identity, d_identity, softmax,
    mse, bce, ce
)
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter
from matplotlib.collections import LineCollection


class SNN():
    """
    Shallow Neural Network:

    One-hidden-layer full-connected neural network for regression or classification (multi-label or multi-class).

    Parameters
    ----------

        input_size : int
            Number of units in the input layer (p).

        output_size : int
            Number of units in the output layer (K).

        hidden_size : int
            Number of hidden units in the hidden layer (M).

        weights: Optional[Union[str, Tuple[np.ndarray, np.ndarray]]], default=None
            Initial weights.
              - None  -> weights are randomly initialized.
              - str   -> path to a '.pkl' file containing a tuple (W, V).
              - tuple -> a pair (W, V) of numpy arrays.
            Expected shapes (bias integrated as a leading column):
              - W: shape (M, p + 1), first column are hidden-layer biases.
              - V: shape (K, M + 1), first column are output-layer biases.

        activation_function : str {'sigmoid', 'relu', 'tanh'}, default='sigmoid'
            Activation function used in the hidden layer.

        task : str {'regression', 'multilabel', 'multiclass'}, default = 'regression'
            Determine the output and loss functions:
            - Regression: identity and MSE
            - Multilabel classification: sigmoid and BCE
            - Multiclass classification: softmax and CE

        optimization_method : str {'SGD', 'Adam'}, default='SGD'
            Method used to update the weights.

        learning_rate : float, default=1e-2
            Base step size to update weights.

        seed : Optional[int], default=None
            Random seed used to initialize the per-instance RNG.

    Public methods
    --------------

        - 'predict(X)'
            return predictions with shape (N, K)

        - 'fit(X, Y, batch_size, epochs, shuffle=True, verbose=False)'
            trains the model and returns a list of epoches losses
        
        - 'save_weights(path)'
            save the tuple (W, V) of matrices in a pickle file
        
        - 'plot_network(ax, title, edge_threshold, max_edge_width, node_size, show_weight_colorbar, show)'
            plots the actual network

    Example
    -------
    
        >>> net = SNN(
        ...     input_size=4, hidden_size=16, output_size=1,
        ...     activation_function='tanh', task='regression'
        ...     optimization_method='Adam', learning_rate=1e-2, 
        ...     seed=25
        ... )
        >>> history = net.fit(X_train, y_train, batch_size=32, epochs=500)
        >>> net.save_weights("weights.pkl")
        >>> y_pred = net.predict(X_test)
        >>> net.plot_network(title="NN after train", edge_threshold=0.05)
    """

    def __init__(
            self,
            input_size: int, 
            output_size: int,
            hidden_size: int,
            weights: Optional[Union[str, Tuple[np.ndarray, np.ndarray]]] = None,
            activation_function: str = "sigmoid",
            task: str = "regression",
            optimization_method: str = "SGD",
            learning_rate: float = 1e-2,
            seed: Optional[int] = None
        ):
        
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        
        self.activation_function = activation_function.lower()
        self._set_activation_function(self.activation_function)

        self.task = task.lower()
        if self.task not in ("regression", "multilabel", "multiclass"):
            raise ValueError(f"Task must be 'regression', 'multilabel' or 'multiclass'")
            
        self._set_output_function()
        self._set_loss_function()

        self.lr = float(learning_rate)

        self.optimization_method = optimization_method.lower()
        if self.optimization_method not in ("sgd", "adam"):
            raise ValueError(f"Optimizer {self.optimization_method} not supported")

        # Initilizate weights
        self._init_weights(weights)

        # Initializate momentums if Adam
        if self.optimization_method == "adam":
            self._init_adam()


    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {X.shape[1]}")
        X = self._add_bias(X)
        return self._forward(X, cache=False)


    def fit(self, X, Y, batch_size: int, epochs: int, shuffle: bool = True, verbose: bool = True):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, got {X.shape[1]}")
        X = self._add_bias(X)

        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, self.output_size)
        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"X and Y must have same number of rows: {X.shape[0]} vs {Y.shape[0]}")
        if Y.shape[1] != self.output_size:
            raise ValueError(f"Y must have {self.output_size} columns, got {Y.shape[1]}")

        N = X.shape[0]
        if batch_size is None or batch_size <= 0 or batch_size > N:
            batch_size = N

        history = []

        for epoch in range(1, epochs + 1):
            ids = np.arange(N)
            if shuffle:
                self.rng.shuffle(ids)

            for start in range(0, N, batch_size):
                batch_ids = ids[start:start + batch_size]
                Xb = X[batch_ids]
                Yb = Y[batch_ids]

                cache = self._forward(Xb, cache=True)
                dW, dV = self._backward(Yb, cache)
                self._apply_update(dW, dV)

            Y_hat = self._forward(X, cache=False)
            loss = self._loss(Y_hat, Y)
            history.append(loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                name = {"regression": "MSE", "multilabel": "BCE", "multiclass": "CE"}[self.task]
                print(f"Epoch {epoch:4d} | {name}: {loss:.6f}")

        return history


    def save_weights(self, path):
        """ Save current weights in a .pkl file """
        try:
            weights = (self.W, self.V)
            with open(path, "wb") as f:
                pickle.dump(weights, f)
        except OSError as e:
            raise ValueError(f"Could not save weights in path '{path}': {e}")


    def _forward(self, X_aug: np.ndarray, cache: bool = False):
        Z = X_aug @ self.W.T # X * W^T size (N x M)
        A = self.sigma(Z) # A = sigma(Z) size (N x M)
        A_aug = self._add_bias(A) # A = (1 | A) size N x (M + 1)
        T = A_aug @ self.V.T # T = A * V^T size N x K
        Y_hat = self.g(T) # Y = g(T) size N x K
        if cache:
            return {"X_aug": X_aug, "Z": Z, "A_aug": A_aug, "T": T, "Y_hat": Y_hat}
        return Y_hat


    def _backward(self, Y: np.ndarray, cache: dict) -> Tuple[np.ndarray, np.ndarray]:
        X_aug = cache["X_aug"]
        Z = cache["Z"]
        A_aug = cache["A_aug"]
        T = cache["T"]
        Y_hat = cache["Y_hat"]

        N, K = Y.shape

        if self.task == "regression":
            # g(T) = (g_1(T_1), ..., g_k(T_K)) 
            deltas = (2.0 / (N * K)) * (Y_hat - Y) * self.d_g(T) # N x K
        else:
            # Multilabel: g(T) = (sigmoid(T_1), ..., sigmoid(T_K))
            # Multiclass: g(T) = softmax(T)
            deltas = (Y_hat - Y) / N
        
        eses = self.d_sigma(Z) * (deltas @ self.V[:, 1:]) # (N x M) * ((N x K) * (K x M))

        dV = deltas.T @ A_aug
        dW = eses.T @ X_aug

        return dW, dV


    def _apply_update(self, dW: np.ndarray, dV: np.ndarray):
        if self.optimization_method == "sgd":
            self.W -= self.lr * dW
            self.V -= self.lr * dV
            return
        elif self.optimization_method == "adam":
            self.r += 1
            b1, b2, eps = self.beta1, self.beta2, self.eps
            # W
            self.mW = b1 * self.mW + (1 - b1) * dW
            self.vW = b2 * self.vW + (1 - b2) * (dW * dW)
            mW_hat = self.mW / (1 - b1**self.r)
            vW_hat = self.vW / (1 - b2**self.r)
            self.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + eps))
            # V
            self.mV = b1 * self.mV + (1 - b1) * dV
            self.vV = b2 * self.vV + (1 - b2) * (dV * dV)
            mV_hat = self.mV / (1 - b1**self.r)
            vV_hat = self.vV / (1 - b2**self.r)
            self.V -= self.lr * (mV_hat / (np.sqrt(vV_hat) + eps))
            return


    def _init_weights(self, weights: Optional[Union[str, Tuple[np.ndarray, np.ndarray]]]) -> None:
        if weights is None:
            self._generate_random_weights()
        elif isinstance(weights, tuple) and len(weights) == 2:
            self._set_weights(weights)
        elif isinstance(weights, str):
            self._load_weights(weights)
        else:
            raise ValueError(f"Weights must be None, a path to a pickle file or a tuple of two np.ndarray")

    
    def _generate_random_weights(self):
        """ Generates random weights based on the activation function """

        match self.activation_function:
            case "sigmoid" | "tanh": # Xavier/Glorot uniform initialization
                limit_W = np.sqrt(6.0 / (self.input_size + self.hidden_size))
                self.W = self.rng.uniform(-limit_W, limit_W, size=(self.hidden_size, self.input_size + 1))
                limit_V = np.sqrt(6.0 / (self.hidden_size + self.output_size))
                self.V = self.rng.uniform(-limit_V, limit_V, size=(self.output_size, self.hidden_size + 1))
            case "relu": # He/Kaiming normal initialization
                std_W = np.sqrt(2.0 / self.input_size)
                self.W = self.rng.normal(0.0, std_W, size=(self.hidden_size, self.input_size + 1))
                std_V = np.sqrt(2.0 / self.hidden_size)
                self.V = self.rng.normal(0.0, std_V, size=(self.output_size, self.hidden_size + 1))
            case _:
                raise ValueError(f"Activation not supported: {self.activation_function}")
        self.W[:, 0] = 0.0
        self.V[:, 0] = 0.0 


    def _set_weights(self, weights):
        """ Set weights from a tuple (W, V) """
        W, V = weights
        if not isinstance(W, np.ndarray) or W.shape != (self.hidden_size, self.input_size + 1):
            raise TypeError(f"W must be a np.ndarray with size {self.hidden_size} x {self.input_size + 1}")
        if not isinstance(V, np.ndarray) or V.shape != (self.output_size, self.hidden_size + 1):
            raise TypeError(f"V must be a np.ndarray with size {self.output_size} x {self.hidden_size + 1}")
        self.W = W
        self.V = V


    def _load_weights(self, path):
        """ Loads weights from a .pkl file """
        try:
            with open(path, "rb") as f:
                loaded_weights = pickle.load(f)
        except OSError as e:
            raise ValueError(f"Could not open weights path '{path}': {e}")
        
        if not (isinstance(loaded_weights, tuple) and len(loaded_weights) == 2):
            raise ValueError("The pickle file must contain a tuple (W, V)")
        W, V = loaded_weights
        if not isinstance(W, np.ndarray) or W.shape != (self.hidden_size, self.input_size + 1):
            raise TypeError(f"W must be a np.ndarray with size {self.hidden_size} x {self.input_size + 1}")
        if not isinstance(V, np.ndarray) or V.shape != (self.output_size, self.hidden_size + 1):
            raise TypeError(f"V must be a np.ndarray with size {self.output_size} x {self.hidden_size + 1}")
        self.W, self.V = W, V


    def _set_activation_function(self, activation_function: str) -> None:
        match activation_function:
            case "sigmoid":
                self.sigma, self.d_sigma = sigmoid, d_sigmoid
            case "relu":
                self.sigma, self.d_sigma = relu, d_relu
            case "tanh":
                self.sigma, self.d_sigma = tanh, d_tanh
            case _:
                raise ValueError(f"Activation function not supported: {activation_function}")


    def _set_output_function(self) -> None:
        match self.task:
            case "regression":
                self.g, self.d_g = identity, d_identity
            case "multilabel":
                self.g, self.d_g = sigmoid, None
            case "multiclass":
                self.g, self.d_g = softmax, None


    def _set_loss_function(self) -> None:
        match self.task:
            case "regression":
                self._loss = mse
            case "multilabel":
                self._loss = bce
            case "multiclass":
                self._loss = ce


    def _init_adam(self):
        # Init momentums
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mV = np.zeros_like(self.V)
        self.vV = np.zeros_like(self.V)

        # Adam hyperparameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps   = 1e-8

        # Count steps
        self.r = 0


    def _add_bias(self, X):
        """ Adds a column of 1s column at the start of a matrix X """
        ones = np.ones((X.shape[0], 1), dtype=X.dtype)
        return np.hstack([ones, X])


    def plot_network(
        self,
        ax: Optional[Axes] = None,
        title: str = "Neural Network",
        edge_threshold: float = 0.0,
        max_edge_width: float = 5.0,
        node_size: float = 600,
        show_weight_colorbar: bool = True,
        show: bool = True
    ) -> Axes:
        """
        Visualize the network on a Matplotlib Axes, where the weights are represented by colors.

        Parameters
        ----------

            ax : Optional[matplotlib.axes.Axes]
                Axes to plot. If None, it creates a new figure.

            title : str
                Title of the plot.

            edge_threshold : float in [0,1)
                Relative threshold: hide edges with |w| < edge_threshold * max(|w|).

            max_edge_width : float
                Maximum width for the edge with the greatest |w|.

            node_size : float
                Size for the nodes (scatter markers).

            show_weight_colorbar : bool
                Show the colorbar for the weights.
            
            show : bool
                If True, the plot is shown.
        """

        N, M, K = self.input_size, self.hidden_size, self.output_size

        # Create the axes
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Positions for the layers
        x_in, x_hid, x_out = 0.0, 1.0, 2.0
        y_in = np.linspace(1.0, 0.0, N) if N > 1 else np.array([0.5])
        y_h  = np.linspace(1.0, 0.0, M) if M > 1 else np.array([0.5])
        y_out = np.linspace(1.0, 0.0, K) if K > 1 else np.array([0.5])
        y_bias_in = 1.15
        y_bias_h  = 1.15

        # Prepare normalization for the colors of the weights
        all_abs = []
        all_abs.append(np.abs(self.W).max())
        all_abs.append(np.abs(self.V).max())
        max_abs = max(all_abs)
        limit = 1e-12
        if max_abs < limit:
            max_abs = limit  # Avoid zero-division
        threshold = edge_threshold * max_abs
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        cmap = get_cmap("coolwarm_r")

        # --- Nodes ---

        # Input nodes (+ bias)
        ax.scatter([x_in] * N, y_in, s=node_size, edgecolor="black", facecolor="white", zorder=3)
        ax.scatter([x_in], [y_bias_in], s=node_size * 0.8, edgecolor="black", facecolor="#eeeeee", zorder=3)
        # Hidden nodes (+ bias)
        ax.scatter([x_hid] * M, y_h, s=node_size * 0.5, edgecolor="black", facecolor="white", zorder=3)
        ax.scatter([x_hid], [y_bias_h], s=node_size * 0.4, edgecolor="black", facecolor="#eeeeee", zorder=3)
        # Output nodes
        ax.scatter([x_out] * K, y_out, s=node_size, edgecolor="black", facecolor="white", zorder=3)

        # --- Edges ---

        segments = []
        colors = []
        widths = []

        # Input bias
        for m in range(M):
            w = self.W[m, 0]
            if abs(w) >= threshold:
                segments.append([(x_in, y_bias_in), (x_hid, y_h[m])])
                colors.append(cmap(norm(w)))
                widths.append(1.0 + (abs(w) / max_abs) * max_edge_width)

        # Input nodes
        for n in range(N):
            for m in range(M):
                w = self.W[m, 1 + n]
                if abs(w) >= threshold:
                    segments.append([(x_in, y_in[n]), (x_hid, y_h[m])])
                    colors.append(cmap(norm(w)))
                    widths.append(1.0 + (abs(w) / max_abs) * max_edge_width)

        # Hidden bias
        for k in range(K):
            v = self.V[k, 0]
            if abs(v) >= threshold:
                segments.append([(x_hid, y_bias_h), (x_out, y_out[k])])
                colors.append(cmap(norm(v)))
                widths.append(1.0 + (abs(v) / max_abs) * max_edge_width)

        # Hidden nodes
        for m in range(M):
            for k in range(K):
                v = self.V[k, 1 + m]
                if abs(v) >= threshold:
                    segments.append([(x_hid, y_h[m]), (x_out, y_out[k])])
                    colors.append(cmap(norm(v)))
                    widths.append(1.0 + (abs(v) / max_abs) * max_edge_width)

        if segments:
            lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.85, zorder=1)
            ax.add_collection(lc)

        # Aesthetic
        ax.set_title(title)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.1, 1.25)
        ax.set_xticks([x_in, x_hid, x_out])
        ax.set_xticklabels(["Input\n(+bias)", "Hidden\n(+bias)", "Output"])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False) # Remove spines

        # Colorbar for the weights
        if show_weight_colorbar:
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fmt2 = FuncFormatter(lambda x, pos: f"{x:.2f}".rstrip("0").rstrip(".")) # Custom formatter: show up to 2 decimals, strip trailing zeros
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, format=fmt2)
            cbar.set_label("Weight values")
            cbar.set_ticks([-max_abs, 0, max_abs])
            
        if show:
            plt.tight_layout()
            plt.show()

        return ax
