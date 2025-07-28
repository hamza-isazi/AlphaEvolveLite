class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.order = []
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.insert(0, key)
        return self.cache[key]

    def put(self, key: int, value: int):
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            old_key = self.order.pop()
            del self.cache[old_key]
        self.cache[key] = value
        self.order.insert(0, key)
