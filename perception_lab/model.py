"""NumPy multinomial logistic regression trained with mini-batch Adam."""

import numpy as np
from .data import features
from .metrics import confusion_matrix, summarize


class PixelClassifier:
    """Small learned baseline; hand-crafted inputs, learned class weights."""

    def __init__(self):
        self.weights = None
        self.mean = None
        self.scale = None

    @staticmethod
    def softmax(logits):
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def transform(self, x):
        return np.column_stack(((x - self.mean) / self.scale, np.ones(len(x))))

    def fit(self, x, y, validation_x, validation_y, epochs=18, batch_size=4096,
            learning_rate=0.025, regularization=0.0005, seed=7):
        if x.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("Invalid training arrays")
        rng = np.random.default_rng(seed)
        # Fit all preprocessing on training data only.
        self.mean = x.mean(axis=0)
        self.scale = np.maximum(x.std(axis=0), 1e-6)
        train, val = self.transform(x), self.transform(validation_x)
        self.weights = np.zeros((train.shape[1], 3))
        first_moment = np.zeros_like(self.weights)
        second_moment = np.zeros_like(self.weights)
        steps, history, best_score, best_weights = 0, [], -1.0, self.weights.copy()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(len(y))
            for start in range(0, len(y), batch_size):
                idx = order[start:start + batch_size]
                probabilities = self.softmax(train[idx] @ self.weights)
                probabilities[np.arange(len(idx)), y[idx]] -= 1
                penalty = self.weights.copy()
                penalty[-1] = 0  # Do not penalize the intercept.
                gradient = train[idx].T @ probabilities / len(idx) + regularization * penalty
                steps += 1
                first_moment = 0.9 * first_moment + 0.1 * gradient
                second_moment = 0.999 * second_moment + 0.001 * gradient**2
                corrected_m = first_moment / (1 - 0.9**steps)
                corrected_v = second_moment / (1 - 0.999**steps)
                self.weights -= learning_rate * corrected_m / (np.sqrt(corrected_v) + 1e-8)
            train_prob = self.softmax(train @ self.weights)
            loss = -np.log(np.clip(train_prob[np.arange(len(y)), y], 1e-12, 1)).mean()
            val_pred = (val @ self.weights).argmax(axis=1)
            score = summarize(confusion_matrix(validation_y, val_pred))["mean_iou"]
            history.append({"epoch": epoch, "training_cross_entropy": float(loss),
                            "validation_mean_iou": score})
            if score > best_score:
                best_score, best_weights = score, self.weights.copy()
                self.best_epoch = epoch
        self.weights = best_weights
        return history

    def predict_features(self, x):
        if self.weights is None:
            raise RuntimeError("Train or load the model first")
        return (self.transform(x) @ self.weights).argmax(axis=1).astype(np.uint8)

    def predict(self, image):
        return self.predict_features(features(image)).reshape(image.shape[:2])

    def save(self, path):
        if self.weights is None:
            raise RuntimeError("Train the model first")
        np.savez_compressed(path, weights=self.weights, mean=self.mean, scale=self.scale,
                            feature_version=np.array([1], dtype=np.int64))

    @classmethod
    def load(cls, path):
        # Numeric .npz only; never unpickle downloaded model objects.
        with np.load(path, allow_pickle=False) as data:
            if data["feature_version"].tolist() != [1]:
                raise ValueError("Unsupported feature version")
            result = cls()
            result.weights, result.mean, result.scale = [data[name].copy()
                                                        for name in ("weights", "mean", "scale")]
        if result.weights.shape != (22, 3) or result.mean.shape != (21,) or result.scale.shape != (21,):
            raise ValueError("Invalid model shapes")
        if (not all(np.isfinite(v).all() for v in (result.weights, result.mean, result.scale))
                or np.any(result.scale <= 0)):
            raise ValueError("Invalid model values")
        return result
