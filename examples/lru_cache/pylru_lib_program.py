import pylru

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = pylru.lrucache(capacity)

    def get(self, key: int) -> int:
        return self.cache[key] if key in self.cache else -1

    def put(self, key: int, value: int):
        self.cache[key] = value
