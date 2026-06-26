"""
SVRFL-style exponential moving average utility score tracker.

Update rule:  u_i = alpha * u_i + (1 - alpha) * sv_i^t

Clients with positive utility are considered trustworthy and may be
used for a filtered aggregation experiment.
"""


class UtilityScoreTracker:
    """Track per-client utility scores across rounds."""

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.scores: dict = {}

    def update(self, shapley_values: dict):
        """Update utility scores with new round Shapley values."""
        for cid, sv in shapley_values.items():
            if cid not in self.scores:
                self.scores[cid] = sv
            else:
                self.scores[cid] = self.alpha * self.scores[cid] + (1 - self.alpha) * sv

    def get_scores(self) -> dict:
        return dict(self.scores)

    def get_positive_clients(self) -> list:
        """Return client IDs with positive utility."""
        return [cid for cid, score in self.scores.items() if score > 0]
