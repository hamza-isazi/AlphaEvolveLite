class LRUCache:
    class _Node:
        __slots__ = 'key', 'value', 'prev', 'next'

        def __init__(self, key, value):
            self.key = key
            self.value = value
            self.prev = None
            self.next = None

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer")
            
        self.capacity = capacity
        self.cache = {}
        
        self.head = self._Node(0, 0)
        self.tail = self._Node(0, 0)
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1

        node.prev.next = node.next
        node.next.prev = node.prev

        node.next = self.head.next
        node.next.prev = node
        self.head.next = node
        node.prev = self.head

        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.cache.get(key)

        if node:
            node.value = value
            node.prev.next = node.next
            node.next.prev = node.prev
            node.next = self.head.next
            node.next.prev = node
            self.head.next = node
            node.prev = self.head
        else:
            if len(self.cache) == self.capacity:
                lru_node = self.tail.prev
                
                lru_node.prev.next = self.tail
                self.tail.prev = lru_node.prev
                
                del self.cache[lru_node.key]

            new_node = self._Node(key, value)
            self.cache[key] = new_node

            new_node.next = self.head.next
            new_node.next.prev = new_node
            self.head.next = new_node
            new_node.prev = self.head