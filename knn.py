import math
from collections import defaultdict

class KNNClassifier:
    def __init__(self, k=3, distance="euclidean", weighted=False):
        self.k = k
        self.distance = distance
        self.weighted = weighted
        self.X = []
        self.y = []

    # -------------------------
    # Training
    # -------------------------
    def fit(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        self.X = X
        self.y = y

    # -------------------------
    # Distance functions
    # -------------------------
    def _euclidean(self, a, b):
        return math.sqrt(
            sum((ai - bi) ** 2 for ai, bi in zip(a, b))
        )

    def _manhattan(self, a, b):
        return sum(abs(ai - bi) for ai, bi in zip(a, b))

    def _distance(self, a, b):
        if self.distance == "euclidean":
            return self._euclidean(a, b)
        elif self.distance == "manhattan":
            return self._manhattan(a, b)
        else:
            raise ValueError("Unsupported distance metric")

    # -------------------------
    # Prediction (single sample)
    # -------------------------
    def predict_one(self, x):
        distances = []

        for xi, label in zip(self.X, self.y):
            d = self._distance(x, xi)

            # Exact match shortcut
            if d == 0:
                return label

            distances.append((d, label))

        # Sort by distance
        distances.sort(key=lambda t: t[0])

        # Take k nearest
        neighbors = distances[:self.k]

        # Option for weight consideration 
        # The closer the neighbour is to the point the more it impacts classification
        if not self.weighted:
            # Majority vote (unweighted)
            votes = {}
            for _, label in neighbors:
                votes[label] = votes.get(label, 0) + 1
            return max(votes, key=votes.get)

        else:
            # Weighted vote: sum(1 / distance)
            weights = defaultdict(float)

            for d, label in neighbors:
                weights[label] += 1.0 / d

            return max(weights, key=weights.get)

    # -------------------------
    # Prediction (batch)
    # -------------------------
    def predict(self, X):
        return [self.predict_one(x) for x in X]
