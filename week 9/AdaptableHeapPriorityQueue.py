import heapq

class AdaptableHeapPriorityQueue:
    def __init__(self):
        self._heap = []  # The actual min-heap (stores (priority, item_id, item_data))
        self._entry_finder = {}  # Maps item_id to its entry in the heap
        self._REMOVED = '<removed-marker>'  # Placeholder for removed items
        self._counter = 0  # Unique sequence count to break ties in priority

    def add_task(self, item_id, item_data, priority):
        if item_id in self._entry_finder:
            self.remove_task(item_id) # If already exists, remove before re-adding with new priority/data
        count = self._counter
        self._counter += 1
        entry = [priority, count, item_id, item_data] # item_id is part of the entry for lookup
        self._entry_finder[item_id] = entry
        heapq.heappush(self._heap, entry)

    def remove_task(self, item_id):
        if item_id in self._entry_finder:
            entry = self._entry_finder.pop(item_id)
            entry[-2] = self._REMOVED # Mark as removed (using item_id position)
            return entry[-1] # Return original item_data
        return None # Or raise an error

    def update_priority(self, item_id, new_priority):
        if item_id in self._entry_finder:
            # To update priority, it's often simplest to remove and re-add
            # More complex implementations might sift up/down directly
            # after locating the item.
            _, _, _, item_data = self._entry_finder[item_id]
            self.remove_task(item_id)
            self.add_task(item_id, item_data, new_priority)
            return True
        return False

    def pop_task(self):
        while self._heap:
            priority, count, item_id, item_data = heapq.heappop(self._heap)
            if item_id is not self._REMOVED:
                if item_id in self._entry_finder: # Check if it wasn't removed externally
                    del self._entry_finder[item_id]
                    return (item_id, item_data, priority)
        raise KeyError('pop from an empty priority queue')

    def is_empty(self):
        # Need to consider removed items still in the heap
        for entry in self._heap:
            if entry[-2] is not self._REMOVED and entry[-2] in self._entry_finder:
                return False
        return True

    def __contains__(self, item_id):
        return item_id in self._entry_finder

    def get_priority(self, item_id):
        if item_id in self._entry_finder:
            return self._entry_finder[item_id][0]
        return None

# Example Usage:
# pq = AdaptableHeapPriorityQueue()
# pq.add_task("task1", {"details": "Urgent task"}, 1)
# pq.add_task("task2", {"details": "Medium task"}, 5)
# pq.add_task("task3", {"details": "Low task"}, 10)

# print(pq.pop_task())  # ('task1', {'details': 'Urgent task'}, 1)

# pq.update_priority("task3", 0) # Make task3 highest priority
# print(pq.pop_task())  # ('task3', {'details': 'Low task'}, 0)

# pq.remove_task("task2")
# print(pq.pop_task()) # Raises KeyError if empty after this or returns next if any