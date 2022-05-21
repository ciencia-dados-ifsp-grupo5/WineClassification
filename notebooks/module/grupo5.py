from typing import Tuple

import numpy as np
from numpy import ndarray

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_X_y


class LogisticRegression(ClassifierMixin, BaseEstimator):
    """
    Logistic Regression multiclass classifier using an One-Vs-Rest strategy.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Learning rate.

    n_epochs : int, default=(1000 | 2 * minibatch_size | 2)
        Number of epochs for training (convergence stop).
        If `optimizer` == 'batch' (default), n_epochs defaults to 1000.
        If `optimizer` == 'minibatch', the default value is 2 * minibatch_size
        If `optimizer` == 'stochastic', the default value is reduced to 2.

    alpha : float, default=0.0001
        Constant that multiplies the regularization term.
        Use 0 to ignore regularization (standard Logistic Regression).
        
    optimizer : {'batch', 'minibatch', 'stochastic'}, defatult='batch'
        Gradient descent optimizer strategy
        
    minibatch_size : int, default=100
        Minibatch size when optimizer` == 'minibatch'

    verbose : int, default=0
        The verbosity level, if non zero, progress messages are printed
        and decision boundary is plot

    random_state : int, default=42
        Seed used for generating random numbers.
        
    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.
    
    coef_ : ndarray of shape (1, n_features) or (n_classes, n_features)
        Coefficient of the features in the decision function.        
        `coef_` is of shape (1, n_features) when the given problem is binary.
        
    intercept_ : ndarray of shape (1,) or (n_classes,)
        Intercept (a.k.a. bias) added to the decision function.
        `intercept_` is of shape (1,) when the given problem is binary.
        
    multilabel_ : bool
        Whether this is a multilabel classifier.
        
    n_classes_ : int
        Number of classes known to the classifier.
        
    n_features_in_ : int
        Number of features seen during `fit`.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = None,
        alpha: float = 0.0001,
        optimizer: str = "batch",
        minibatch_size: int = 100,
        verbose: int = 0,
        random_state: int = 42
    ):

        assert isinstance(learning_rate, (float, int)) and (learning_rate > 0), \
        f'Learning rate must be a `float` > 0.0. Passed: "{learning_rate}"'
        
        assert (n_epochs is None) or (isinstance(n_epochs, int) and (n_epochs > 0)), \
        f'Number of epochs must be an `int` > 0. Passed: "{n_epochs}"'
        
        assert isinstance(alpha, (float, int)) and (alpha >= 0), \
        f'Alpha must be a `float` >= 0.0. Passed: "{alpha}"'
        
        assert optimizer in ['batch', 'minibatch', 'stochastic'], \
        f'Optimizer must be in {"batch", "minibatch", "stochastic"}. Passed: "{optimizer}"'
        
        assert (optimizer != 'minibatch') or (isinstance(minibatch_size, int) and (minibatch_size > 0)), \
        f'Minibatch size must be an `int` > 0. Passed: "{minibatch_size}"'
        
        if n_epochs is None:
            if optimizer == 'minibatch':
                n_epochs = 2 * minibatch_size
            elif optimizer == 'stochastic':
                n_epochs = 2
            else:
                n_epochs = 1000
        
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.alpha = alpha
        self.optimizer = optimizer
        self.minibatch_size = minibatch_size
        self.verbose = verbose
        self.random_state = random_state
        
        # parameters to be trained/learned
        self.__w = None  # weight array
        self.__b = None  # bias
        
        self.__classes = None
    
    
    # a special method used to represent a class object as a string, called with print() or str()
    def __str__(self):
        msg = (
            "Multiclass Logistic Regressor Instance\n\n"
            f"Learning rate: {self.learning_rate}\n"
            f"Number of epochs: {self.n_epochs}\n"
            f"Regularization constant (alpha): {self.alpha}\n"
            f"Optimizer (optimizer): {self.optimizer}\n"
            f"{f'Minibatch size: {self.minibatch_size}' if self.optimizer == 'minibatch' else ''}\n"
            f"Verbose: {self.verbose}\n"
            f"Random state: {self.random_state}\n\n"
            f"Trained?: {self.is_fitted()}"
        )

        return msg
    
    @property
    def classes_(self) -> ndarray:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__classes
    
    
    @property
    def n_classes_(self) -> int:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return len(self.__classes)
    
    
    @property
    def multilabel_(self) -> int:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.n_classes_ > 2
    
    
    @property
    def coef_(self) -> ndarray:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__w
    

    @property
    def intercept_(self) -> ndarray:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__b
    
    
    @property
    def n_features_in_(self) -> int:
        assert self.is_fitted(), 'The instance is not fitted yet.'
        return self.__w.shape[1]
    
    
    def is_fitted(self) -> bool:
        return self.__w is not None
    
    
    def __sigmoid(self, z: ndarray) -> ndarray:
        return 1 / (1 + np.e ** (-z))
    
        
    def __shuffle(self, X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Return a random permutation of features and labels
        """
        p = np.random.permutation(len(y))
        return X[p], y[p]
    
    
    def __log_loss(self, y: ndarray, p_hat: ndarray, eps: float = 1e-15) -> float:
        """
        Return the log loss for a given estimation and ground-truth (true labels).
        
        log is undefined for 0. Consequently, the log loss is undefined for
        `p_hat=0` (because of log(p_hat)) and `p_hat=1` (because of ln(1 - p_hat)).
        
        To overcome that, we clipped the probabilities to max(eps, min(1 - eps, p_hat)),
        where `eps` is a tiny constant. 

        Parameters
        ----------
        y : ndarray, of shape (n_samples,)
            True labels of input samples.
            
        p_hat : ndarray
            Estimated probabilities of input samples.
            
        eps : float, default=1e-15
            Epsilon term used to avoid undefined log loss at 0 and 1.
        
        Returns
        -------
        log_loss : float
            Computed log loss.
        """
        
        p_hat_eps = np.maximum(eps, np.minimum(1 - eps, p_hat))
        
        losses = -(y * np.log(p_hat_eps) + (1 - y) * np.log(1 - p_hat_eps))
        log_loss = losses.mean()
        
        return log_loss
    
    
    def __gradient(
        self,
        X: ndarray,
        y: ndarray,
        p_hat: ndarray,
        w: ndarray,
        alpha: float
    ) -> Tuple[ndarray, float]:
        '''
        Compute the gradient vector for the log loss with regards to the weights and bias.
        
        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,).
            Target (true) labels.
        p_hat : ndarray of shape (n_samples,)
            Estimated probabilities.
        w : ndarray of shape (n_features,)
            Weight array.
        alpha : float
            Reguralization constant.
        
        Returns
        -------
        Tuple[ndarray, float]: 
            Tuple with:
            - a numpy array of shape (n_features,) containing the partial derivatives w.r.t. the weights; and
            - a float representing the partial derivative w.r.t. the bias.
        '''
        
        n_samples = len(X)
        
        regularization = alpha * w
        
        error = p_hat - y
        grad_w = (np.dot(error, X) / n_samples) + regularization
        grad_b = error.mean()
        
        return grad_w, grad_b

    
    def __plot_boundaries(self, X: ndarray, y: ndarray, weights: ndarray, biases: ndarray, verbose: int):      
        fig = plt.figure(figsize=(16,8), tight_layout=True)
        gs = GridSpec(3, 3, figure=fig)
            
        ax1 = fig.add_subplot(gs[:, :-1])
        ax2 = fig.add_subplot(gs[0, -1])
        ax3 = fig.add_subplot(gs[1, -1])
        ax4 = fig.add_subplot(gs[2, -1])
            
        w_lines = weights[:-1:verbose] + [ weights[-1] ]
        b_lines = biases[:-1:verbose] + [ biases[-1] ]
        colors = sns.color_palette("crest", n_colors=len(w_lines))
        
        for i, (w1, w2), b in zip(range(len(w_lines)), w_lines, b_lines):
            x1_decision_line = np.array([X[:, 0].min(), X[:, 0].max()])
            x2_decision_line = -(b + (w1 * x1_decision_line)) / w2
            sns.lineplot(x=x1_decision_line, y=x2_decision_line, color=colors[i], ax=ax1)
                
        sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, ax=ax1, zorder=99)
                
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_xlim(X[:, 0].min(), X[:, 0].max())
        ax1.set_ylim(X[:, 1].min(), X[:, 1].max())
        ax1.set_title('Updates of Decision Boundary on Training Samples')
            
        weights = np.array(weights)
        biases = np.array(biases)
        sns.lineplot(x=weights[:, 0], y=biases, ax=ax2)
        ax2.set(title="Evolution of w1 x b", xlabel="w1", ylabel="b")
            
        sns.lineplot(x=weights[:, 1], y=biases, ax=ax3)
        ax3.set(title="Evolution of w2 x b", xlabel="w2", ylabel="b")
            
        sns.lineplot(x=weights[:, 0], y=weights[:, 1], ax=ax4)
        ax4.set(title="Evolution of w1 x w2", xlabel="w1", ylabel="w2")
            
        plt.show()

    
    def __binary_fit(self, X: ndarray, y: ndarray):
        '''
        Train a Binary Logistic Regression classifier.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,)
            Target (true) labels.
            
        Returns
        -------
        w, b: Tuple[ndarray, float]
            Weights (n_features,) and bias learned from the fit method
        '''

        n_samples, n_features = X.shape

        ### PARAMETER INITIALIZATION
        # return values from the "standard normal" distribution.
        w = np.random.randn(n_features)
        b = np.random.randn()
        
        # values of each iteration
        losses = []
        weights = [w]
        biases = [b]
        
        if self.optimizer == 'stochastic':
            batch_size = 1
        elif self.optimizer == 'minibatch':
            batch_size = self.minibatch_size
        else:
            batch_size = n_samples
        
        # LEARNING ITERATIONS
        batch_range = range(0, n_samples, batch_size)
        total_iterations = self.n_epochs * len(batch_range)
        n_iteration = 0
        
        for epoch in range(self.n_epochs):
            X, y = self.__shuffle(X, y)
            for i in batch_range:
                X_b = X[i:i + batch_size]
                y_b = y[i:i + batch_size]
                
                ### ESTIMATION (FORWARD PASS)
                z = np.dot(X_b, w) + b
                p_hat = self.__sigmoid(z)
            
                ### LOSS
                J = self.__log_loss(y_b, p_hat)

                losses.append(J)
                
                ### GRADIENT DESCENT UPDATES (BACKWARD PASS)
                grad_w, grad_b = self.__gradient(X_b, y_b, p_hat, w, self.alpha)
                w = w - self.learning_rate * grad_w
                b = b - self.learning_rate * grad_b
                
                weights.append(w)
                biases.append(b)
                
                n_iteration += 1
                if self.verbose > 0 and ((n_iteration == 1) or (n_iteration % self.verbose == 0)) :
                    print(f'[INFO] iteration={n_iteration}/{total_iterations}, loss={J:.7f}')
                                    
        if self.verbose > 0:
            losses = np.array(losses)
            print(f'\nFinal loss: {losses[-1]}')
            print(f'\nMean loss: {losses.mean()} +- {losses.std()}')
            self.__plot_boundaries(X, y, weights, biases, self.verbose)
            
        return weights[-1], biases[-1]
            
    
    def fit(self, X: ndarray, y: ndarray):
        '''
        Train a Binary or Multiclass Logistic Regression classifier.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data.
        y: ndarray of shape (n_samples,)
            Target (true) labels.
            
        Returns
        -------
        self : object
            Returns self.
        '''          

        # Validate X and y
        X, y = check_X_y(X, y)

        np.random.seed(self.random_state)
        
        self.__classes = np.unique(y)
        
        # "Special" case when we have only 2 classes (train just one classifier, not 2)
        classes = self.__classes if len(self.__classes) > 2 else self.__classes[-1:]
        
        ### ONE-VS-REST
        
        # w: (n_classes, n_features) if multiclass OR (1, n_features) if binary
        # b: (n_classes,) if multiclass OR (1,) if binary
        w, b = [], []
        
        # For each class, train a binary classifier
        for label in classes:
            # The current label is set to the positive class (1)
            # and all the others, to the negative one (0).
            y_label = np.array([1 if l == label else 0 for l in y])
            
            # Tain a binary classifier considering the current binary labels
            if self.verbose and len(self.__classes) > 2:
                print(f'### TRAINING FOR CLASS "{label}" ###')
            w_label, b_label = self.__binary_fit(X, y_label)
            
            w.append(w_label)
            b.append(b_label)
            
        self.__w = np.array(w)
        self.__b = np.array(b)
            
        return self
    
    
    def predict_proba(self, X: ndarray) -> ndarray:
        '''
        Estimate the probability for each class of input samples.

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            The estimated probabilities for each class of input samples.
        '''
        assert self.is_fitted(), 'The instance is not fitted yet.'
        assert X.ndim == 2, f'X must b 2D. Passed: {X.ndim}'

        z = np.dot(X, self.__w.T) + self.__b
        p_hat = self.__sigmoid(z)
        
        return p_hat  
        
        
    def predict(self, X: ndarray) -> ndarray:
        '''
        Predict the labels for input samples.
        
        If a binary problem, thresholding at probability >= 0.5 to positive class.
        If a multiclass problem, return the class with max probability

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted labels of input samples.
        '''
        assert self.is_fitted(), 'The instance is not fitted yet.'
        assert X.ndim == 2, f'X must b 2D. Passed: {X.ndim}'
        
        p_hat = self.predict_proba(X)
        
        y_hat = np.argmax(p_hat, axis=1) if self.multilabel_ else (p_hat[:, 0] >= 0.5).astype(int)
        
        return self.classes_[y_hat]
