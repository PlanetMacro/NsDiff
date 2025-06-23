import numpy as np
class CRPS:
    def __init__(self, preds, truth):
        self.preds=preds; self.truth=truth
    def compute(self):
        return [float(np.mean(np.abs(self.preds-self.truth)))]
