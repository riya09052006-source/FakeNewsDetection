from collections import OrderedDict
from typing import Optional


class PredictionCache:

    def __init__(
        self,
        max_size: int = 100,
    ):
        self.max_size = max_size

        self.cache = OrderedDict()

    def get(
        self,
        key: str,
    ) -> Optional[dict]:

        if key not in self.cache:
            return None

        value = self.cache.pop(key)

        self.cache[key] = value

        return value

    def set(
        self,
        key: str,
        value: dict,
    ):

        if key in self.cache:
            self.cache.pop(key)

        self.cache[key] = value

        while len(self.cache) > self.max_size:
            self.cache.popitem(
                last=False
            )

    def clear(self):

        self.cache.clear()

    def size(self):

        return len(self.cache)


prediction_cache = PredictionCache(
    max_size=100
)
