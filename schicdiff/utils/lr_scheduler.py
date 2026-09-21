import math


class CosineLambdaSchedule:
    def __init__(self, max_steps, min_factor=0.0):
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}")
        if not 0.0 <= min_factor <= 1.0:
            raise ValueError(f"min_factor must be in [0, 1], got {min_factor}")
        self.max_steps = max_steps
        self.min_factor = min_factor

    def schedule(self, step):
        progress = min(max(step, 0), self.max_steps) / self.max_steps
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_factor + (1.0 - self.min_factor) * cosine
