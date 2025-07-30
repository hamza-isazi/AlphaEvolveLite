class _Node:
    __slots__ = 'key', 'value', 'prev', 'next'

    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}  # Stores key -> _Node object
        self.capacity = capacity
        self.size = 0    # Track current size of cache

        # Dummy head and tail nodes to simplify edge cases
        self.head = _Node()
        self.tail = _Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1

        # Move node to front: Unlink
        node.prev.next = node.next
        node.next.prev = node.prev

        # Move node to front: Relink at head
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

        return node.value

    def put(self, key: int, value: int):
        if self.capacity <= 0:
            return

        node = self.cache.get(key)
        if node:
            # Key exists: unlink it to move to front.
            node.prev.next = node.next
            node.next.prev = node.prev
        else:
            # New key.
            if self.size < self.capacity:
                # Not full: create a new node, increment size.
                node = _Node()
                self.size += 1
            else:
                # Full: evict LRU node (tail.prev) and reuse it.
                node = self.tail.prev
                node.prev.next = self.tail
                self.tail.prev = node.prev
                del self.cache[node.key]

            # Set new key for the node and add to cache.
            node.key = key
            self.cache[key] = node

        # For all cases, update value and link node to the front.
        node.value = value
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node
