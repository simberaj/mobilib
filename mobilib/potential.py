
import numpy as np

class DecayFunction:
    def __init__(self, fullrange: float):
        self.fullrange = fullrange
    
    def _correct_dist_one(self, distance: float):
        if self.fullrange == 0:
            return distance
        else:
            raw = distance - self.fullrange
            return np.where(raw < 0, 0, raw)
    
    @classmethod
    def create(cls, name, *args, **kwargs):
        return cls.FXS[name](*args, **kwargs)


class GaussianDecay(DecayFunction):
    def __init__(self, halfrange: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.halfrange = halfrange
        self.halfrange_sq = halfrange ** 2
    
    def decay(self, distance):
        return 2 ** (-self._correct_dist(distance) ** 2 / self.halfrange_sq)
        

class LinearDecay(DecayFunction):
    def __init__(self, halfrange: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.halfrange = halfrange
    
    def decay(self, distance):
        decayed = 1 - self._correct_dist(distance) / (2 * halfrange)
        return np.where(decayed < 0, 0, decayed)

DecayFunction.FXS = {
    'gaussian': GaussianDecay,
    'linear': LinearDecay,
}