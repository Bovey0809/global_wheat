from collections import defaultdict


class EarlyStop(object):
    """Used for early stopping.

    Early stop based on steps.

    Attributes:
        steps: after N steps no changes then stop.
    """

    def __init__(self, steps):
        self.steps = steps
        self.maximum = float('-inf')
        self.losses = []

    def add(self, item):
        if len(self.losses) > self.steps:
            self.losses.pop(0)
