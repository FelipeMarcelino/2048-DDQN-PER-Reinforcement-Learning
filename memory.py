import numpy as np
from collections import namedtuple
from numba import jitclass, int64, float64, bool_


spec = [
    ("__capacity", int64),
    ("__data_pointer", int64),
    ("__tree", float64[:]),
    ("__state", float64[:, :, :, :]),
    ("__action", int64[:]),
    ("__reward", float64[:]),
    ("__next_state", float64[:, :, :, :]),
    ("__done", bool_[:]),
]


@jitclass(spec)
class SumTree:
    def __init__(self, capacity, size_board=4):
        # Pointer to leaf tree
        self.__data_pointer = 0

        # Numbers of leaf nodes that contains experience
        self.__capacity = capacity

        # Initialize the tree with all nodes equal zero
        # Leaf nodes = capacity
        # Parent nodes = capacity - 1(minus root)
        # Priority tree = 2 * capacity - 1
        self.__tree = np.zeros(2 * capacity - 1)

        # Initialize experience tree with zeros
        self.__state = np.zeros((capacity, size_board, size_board, 16))
        self.__action = np.zeros(capacity, dtype=np.int64)
        self.__reward = np.zeros(capacity)
        self.__next_state = np.zeros((capacity, size_board, size_board, 16))
        self.__done = np.zeros(capacity, dtype=np.bool_)

    def __update(self, tree_index, priority):
        # Change new priority score - former priority score
        change = priority - self.__tree[tree_index]
        self.__tree[tree_index] = priority

        # Propagate changes through tree and change the parents
        while tree_index != 0:
            tree_index = tree_index - 1 // 2  # Round the result to index
            self.__tree[tree_index] += change

    def add(self, priority, state, action, reward, next_state, done):
        """Add experience in replay store memory"""

        # Put data inside arrays
        self.__state[self.__data_pointer] = state
        self.__action[self.__data_pointer] = action
        self.__reward[self.__data_pointer] = reward
        self.__next_state[self.__data_pointer] = next_state
        self.__done[self.__data_pointer] = done

        # Update prioritized tree. Obs: Fill the leaves from left to right
        tree_index = self.__data_pointer + self.__capacity - 1
        self.__update(tree_index, priority)

        # Change the data pointer to next leaf
        self.__data_pointer += 1

        # Check if data pointer reaches the maximum capacity, than back to the first index and overwrite data
        if self.__data_pointer >= self.__capacity:
            self.__data_pointer = 0

    def get_leaf(self, value):
        """Retrieve one leaf priority and the data's leaf"""
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.__tree):
                leaf_index = parent_index
                break

            else:  # downward search, always search for a higher priority node

                if value <= self.__tree[left_child_index]:
                    parent_index = left_child_index

                else:
                    value -= self.__tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.__capacity + 1

        return (
            leaf_index,
            self.__tree[leaf_index],
            self.__state[data_index],
            self.__action[data_index],
            self.__reward[data_index],
            self.__next_state[data_index],
            self.__done[data_index],
        )

    @property
    def total_priority(self):
        """Get total priority's tree"""
        return self.__tree[0]
