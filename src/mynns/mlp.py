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


class MLP():
    """
    Multilayer Perceptron:

    Full-connected neural network with multiple hidden layers.

    Parameters
    ----------

        layer_sizes : list[int]
            List with the number of units for each layer:
            [M_{0}, M_{1}, ..., M_{l}, ..., M_{L}, M_{L+1}]
        
        weights : Optional[Union[str, Tuple[list[np.ndarray], list[np.ndarray]]]
            Initial weights.
              - None  -> weights are randomly initialized.
              - str   -> path to a '.pkl' file containing a tuple (W, b).
              - tuple -> a pair (W, b) of lists of numpy arrays.
            Expected shapes:
              - W: shape (L+1, M_{l+1}, M_{l})  
              - b: shape (L+1, 1, M_{l+1})
            
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
        
        l2_lambda : float, default=0.0
            Parameter for L2 regularization.

        seed : Optional[int], default=None
            Random seed used to initialize the per-instance RNG.

    Public methods
    --------------

        - 'predict(X)'
            return predictions with shape (N, K)

        - 'fit(X, Y, batch_size, epochs, shuffle=True, verbose=False)'
            trains the model and returns a list of epochs losses
        
        - 'save_weights(path)'
            save the tuple (W, b) of np.arrays in a pickle file

    Example
    -------

        >>> net = MLP(
        ...     layer_sizes=[8, 64, 32, 16, 2]
        ...     activation_function='tanh', task='classification'
        ...     optimization_method='Adam', learning_rate=1e-2, 
        ...     seed=25
        ... )
        >>> history = net.fit(X_train, y_train, batch_size=32, epochs=500)
        >>> net.save_weights("weights.pkl")
        >>> y_pred = net.predict(X_test)
    """

    def __init__(
            self,
            layer_sizes: list[int],
            weights: Optional[Union[str, Tuple[list[np.ndarray], list[np.ndarray]]]] = None,
            activation_function: str = "sigmoid",
            task: str = "regression",
            optimization_method: str = "SGD",
            learning_rate: float = 1e-2,
            l2_lambda: float = 0.0,
            seed: Optional[int] = None
        ):

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self._set_layers_sizes(layer_sizes)
        self._set_activation_function(activation_function)

        self.task = task.lower()
        if self.task not in ("regression", "multilabel", "multiclass"):
            raise ValueError(f"Task must be 'regression', 'multilabel' or 'multiclass'")
            
        self._set_output_function()
        self._set_loss_function()

        self._set_optimization_method(optimization_method)
        self.lr = learning_rate
        self.l2_lambda = l2_lambda

        self._init_weights(weights)
        if self.optimization_method == "adam":
            self._init_adam()

        
    def fit(self, X, Y, batch_size: int, epochs: int, shuffle: bool = True, verbose: bool = True):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")

        Y = np.asarray(Y, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, self.K)
        if Y.shape[1] != self.K:
            raise ValueError(f"Expected {self.K} columns, got {Y.shape[1]}")

        if Y.shape[0] != X.shape[0]:
            raise ValueError(f"X and Y must have same number of rows: {X.shape[0]} vs {Y.shape[0]}")

        N = X.shape[0]
        if batch_size <= 0 or batch_size > N:
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

                A, Z = self._forward(Xb)
                dW, db = self._backward(A, Z, Yb)
                self._apply_update(dW, db)

            Y_hat = self._forward(X)[0][-1]
            data_loss = self._loss(Y_hat, Y)
            loss = data_loss + 0.5 * self.l2_lambda * self._l2_penalty()
            history.append(loss)

            if verbose and (epoch == 1 or epoch % max(1, epochs // 10) == 0 or epoch == epochs):
                name = {"regression": "MSE", "multilabel": "BCE", "multiclass": "CE"}[self.task]
                if self.l2_lambda > 0.0:
                    name += "+L2"
                print(f"Epoch {epoch:4d} | {name}: {loss:.6f}")

        return history


    def save_weights(self, path: str) -> None:
        """ Save current weights in a .pkl file """
        try:
            weights = (self.W, self.b)
            with open(path, "wb") as f:
                pickle.dump(weights, f)
        except OSError as e:
            raise ValueError(f"Could not save weights in path '{path}': {e}")


    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != self.p:
            raise ValueError(f"Expected {self.p} features, got {X.shape[1]}")
        return self._forward(X)[0][-1]


    def _forward(self, X: np.ndarray) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        A, Z = [X], []
        for l in range(self.L):
            Z.append(A[-1] @ self.W[l].T + self.b[l])
            A.append(self.sigma(Z[-1]))
        Z.append(A[-1] @ self.W[-1].T + self.b[-1])
        A.append(self.g(Z[-1])) # Y
        return A, Z


    def _backward(self, A: list[np.ndarray], Z: list[np.ndarray], Y: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        N, K = Y.shape
        dW = [np.zeros_like(Wl) for Wl in self.W]
        db = [np.zeros_like(bl) for bl in self.b]

        if self.task == "regression":
            deltas = (2.0 / (N * K)) * (A[-1] - Y) * self.d_g(Z[-1])
        else:
            deltas = (A[-1] - Y) / N

        for l in reversed(range(self.L + 1)):
            dW[l] = deltas.T @ A[l] # (N x M_{l+1})^T x (N x M_{l})
            db[l] = np.sum(deltas, axis=0, keepdims=True) # (1 x M_{l+1})
            if l > 0:
                deltas = (deltas @ self.W[l]) * self.d_sigma(Z[l-1]) # (N x M_{l+1}) x (M_{l+1} x M_{l})

        if self.l2_lambda > 0.0:
            for l in range(self.L + 1):
                dW[l] += self.l2_lambda * self.W[l]

        return dW, db


    def _apply_update(self, dW: list[np.ndarray], db: list[np.ndarray]) -> None:
        if self.optimization_method == "sgd":
            for l in range(self.L + 1):
                self.W[l] -= self.lr * dW[l]
                self.b[l] -= self.lr * db[l]
            return
        elif self.optimization_method == "adam":
            self.r += 1
            b1, b2, eps = self.beta1, self.beta2, self.eps
            for l in range(self.L + 1):
                # W
                self.mW[l] = b1 * self.mW[l] + (1 - b1) * dW[l]
                self.vW[l] = b2 * self.vW[l] + (1 - b2) * (dW[l] * dW[l])
                mW_hat = self.mW[l] / (1 - b1**self.r)
                vW_hat = self.vW[l] / (1 - b2**self.r)
                self.W[l] -= self.lr * (mW_hat / (np.sqrt(vW_hat) + eps))
                # b
                self.mb[l] = b1 * self.mb[l] + (1 - b1) * db[l]
                self.vb[l] = b2 * self.vb[l] + (1 - b2) * (db[l] * db[l])
                mb_hat = self.mb[l] / (1 - b1**self.r)
                vb_hat = self.vb[l] / (1 - b2**self.r)
                self.b[l] -= self.lr * (mb_hat / (np.sqrt(vb_hat) + eps))
            return


    def _init_adam(self):
        # Init momentums
        self.mW = [np.zeros_like(Wl) for Wl in self.W]
        self.vW = [np.zeros_like(Wl) for Wl in self.W]
        self.mb = [np.zeros_like(bl) for bl in self.b]
        self.vb = [np.zeros_like(bl) for bl in self.b]
        # Adam hyperparameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps   = 1e-8
        # Count steps
        self.r = 0


    def _init_weights(self, weights: Optional[Union[str, Tuple[list[np.ndarray], list[np.ndarray]]]]) -> None:
        if weights is None:
            self._generate_random_weights()
        elif isinstance(weights, str):
            self._load_weights(weights)
        elif isinstance(weights, tuple) and len(weights) == 2:
            self._set_weights(weights)
        else:
            raise ValueError(f"Weights must be None, a path to a pickle file or a tuple (W, b) of lists of np.ndarray")


    def _generate_random_weights(self) -> None:
        """ Generates random weights based on the activation function """
        match self.activation_function:
            case "sigmoid" | "tanh": # Xavier/Glorot uniform initialization
                limits_W = [np.sqrt(6.0 / (self.layer_sizes[l] + self.layer_sizes[l+1])) for l in range(self.L + 1)]
                self.W = [self.rng.uniform(-limits_W[l], limits_W[l], size=(self.layer_sizes[l+1], self.layer_sizes[l])) for l in range(self.L + 1)]
            case "relu": # He/Kaiming normal initialization
                stds_W = [np.sqrt(2.0 / Ml) for Ml in self.layer_sizes[:-1]]
                self.W = [self.rng.normal(0.0, stds_W[l], size=(self.layer_sizes[l+1], self.layer_sizes[l])) for l in range(self.L + 1)]
        # Bias always initialized at 0
        self.b = [np.zeros(shape=(1, Ml)) for Ml in self.layer_sizes[1:]]


    def _load_weights(self, path: str) -> None:
        """ Loads weights from a .pkl file """
        try:
            with open(path, "rb") as f:
                loaded_weights = pickle.load(f)
        except OSError as e:
            raise ValueError(f"Could not open weights path '{path}': {e}")
        
        if not (isinstance(loaded_weights, tuple) and len(loaded_weights) == 2):
            raise ValueError("The pickle file must contain a tuple (W, b)")

        self._set_weights(loaded_weights)


    def _set_weights(self, weights: Tuple[list[np.ndarray], list[np.ndarray]]) -> None:
        """ Set weights from a tuple (W, b) """
        W, b = weights
        self.W = [np.asarray(Wl, dtype=float) for Wl in W]
        self.b = [np.asarray(bl, dtype=float) for bl in b]

        self._validate_weights_shapes()


    def _validate_weights_shapes(self) -> None:
        M = self.layer_sizes
        if len(self.W) != self.L + 1 or len(self.b) != self.L + 1:
            raise ValueError("W and b must have length L+1.")
        for l in range(self.L + 1):
            Wl, bl = self.W[l], self.b[l]
            if Wl.shape != (M[l+1], M[l]):
                raise ValueError(f"W[{l}] shape {Wl.shape} != {(M[l+1], M[l])}")
            if bl.shape != (1, M[l+1]):
                raise ValueError(f"b[{l}] shape {bl.shape} != {(1, M[l+1])}")


    def _set_layers_sizes(self, layer_sizes: list[int]) -> None:
        if (L := len(layer_sizes)) < 3:
            raise ValueError("There must be at least 1 hidden layer: 'layer_size' must have at least length 3.")
        self.p = layer_sizes[0]
        self.L = L - 2
        self.K = layer_sizes[-1]
        self.layer_sizes = layer_sizes


    def _set_activation_function(self, activation_function: str) -> None:
        self.activation_function = activation_function.lower()
        match self.activation_function:
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


    def _set_optimization_method(self, optimization_method: str) -> None:
        self.optimization_method = optimization_method.lower()
        if self.optimization_method not in ["sgd", "adam"]:
            raise ValueError(f"Optimizer not supported: {self.optimization_method}")


    def _l2_penalty(self) -> float:
        return sum(np.sum(Wl * Wl) for Wl in self.W)


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

        # Create the axes
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 6))

        # Positions for the layers
        x_positions = np.arange(self.L + 2, dtype=float)
        y_positions = []
        for m in self.layer_sizes:
            if m > 1:
                y_positions.append(np.linspace(1.0, 0.0, m))
            else:
                y_positions.append(np.array([0.5]))
        x_positions_bias = x_positions + 0.15
        y_positions_bias = 1.15

        # Prepare normalization for the colors of the weights
        all_abs = []
        for l in range(self.L + 1):
            all_abs.append(np.abs(self.W[l]).max())
            all_abs.append(np.abs(self.b[l]).max())
        max_abs = max(all_abs)
        limit = 1e-12
        if max_abs < limit:
            max_abs = limit  # Avoid zero-division
        threshold = edge_threshold * max_abs
        norm = Normalize(vmin=-max_abs, vmax=max_abs)
        cmap = get_cmap("coolwarm_r")

        # --- Nodes ---

        hidden_scale = 0.6
        bias_scale = 0.6

        # Input nodes
        ax.scatter([x_positions[0]] * self.layer_sizes[0], y_positions[0],
            s=node_size, edgecolor="black", facecolor="white", zorder=3)

        # Hidden nodes (+ bias)
        for l in range(1, self.L + 1):
            ax.scatter(
                [x_positions[l]] * self.layer_sizes[l], y_positions[l],
                s=node_size * hidden_scale, edgecolor="black", facecolor="white", zorder=3)

        # Output nodes
        ax.scatter(
            [x_positions[-1]] * self.layer_sizes[-1], y_positions[-1],
            s=node_size, edgecolor="black", facecolor="white", zorder=3)

        # Biases nodes
        for l in range(self.L + 1):  # 0..L
            ax.scatter(
                [x_positions_bias[l]], [y_positions_bias],
                s=node_size * bias_scale, edgecolor="black", facecolor="#eeeeee", zorder=3)

        # --- Edges ---

        segments = []
        colors = []
        widths = []

        for l in range(self.L + 1):  # 0..L
            x_left = x_positions[l]
            x_right = x_positions[l + 1]
            y_left = y_positions[l]
            y_right = y_positions[l + 1]

            # 1) From bias of layer l to nodes of layer l+1 (weights b[l])
            bl = self.b[l].reshape(-1)  # (M_{l+1},)
            for j in range(self.layer_sizes[l + 1]):
                w = bl[j]
                if abs(w) >= threshold:
                    segments.append([(x_positions_bias[l], y_positions_bias), (x_right, y_right[j])])
                    colors.append(cmap(norm(w)))
                    widths.append(1.0 + (abs(w) / max_abs) * max_edge_width)

            # 2) From nodes of layer l to nodes of layer l+1 (weights W[l])
            Wl = self.W[l]
            for i in range(self.layer_sizes[l]):
                for j in range(self.layer_sizes[l + 1]):
                    w = Wl[j, i]
                    if abs(w) >= threshold:
                        segments.append([(x_left, y_left[i]), (x_right, y_right[j])])
                        colors.append(cmap(norm(w)))
                        widths.append(1.0 + (abs(w) / max_abs) * max_edge_width)

        if segments:
            edges = list(zip(widths, segments, colors))
            edges.sort(key=lambda x: x[0])  # Sort by widths
            widths_sorted, segments_sorted, colors_sorted = zip(*edges)
            lc = LineCollection(segments_sorted, colors=colors_sorted, linewidths=widths_sorted, alpha=0.85, zorder=1)
            ax.add_collection(lc)

        # --- Aesthetic ---
        ax.set_title(title)
        ax.set_xlim(-0.5, self.L + 1.5)
        ax.set_ylim(-0.1, 1.25)
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False) # Remove spines

        # Etiquetas del eje X
        xticklabels = []
        for l in range(self.L + 2):
            if l == 0:
                xticklabels.append("Input\n(+bias)")
            elif l == self.L + 1:
                xticklabels.append("Output")
            else:
                xticklabels.append(f"Hidden {l}\n(+bias)")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticklabels)

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
