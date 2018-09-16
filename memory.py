import numpy as np
import ipdb
from collections import namedtuple
from numba import jitclass, int64, float64, bool_, jit, njit


spec_sum_tree = [
    ("__capacity", int64),
    ("__data_pointer", int64),
    ("__tree", float64[:]),
    ("__state", float64[:, :, :, :, :]),
    ("__action", int64[:]),
    ("__reward", float64[:]),
    ("__next_state", float64[:, :, :, :, :]),
    ("__done", bool_[:]),
]


@jitclass(spec_sum_tree)
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
        self.__state = np.zeros((capacity, 1, size_board, size_board, 16))
        self.__action = np.zeros(capacity, dtype=np.int64)
        self.__reward = np.zeros(capacity)
        self.__next_state = np.zeros((capacity, 1, size_board, size_board, 16))
        self.__done = np.zeros(capacity, dtype=np.bool_)

    def update(self, tree_index, priority):
        # Change new priority score - former priority score
        change = priority - self.__tree[tree_index]
        self.__tree[tree_index] = priority

        # Propagate changes through tree and change the parents
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2  # Round the result to index
            self.__tree[tree_index] += change

    def add(self, priority, state, action, reward, next_state, done):
        """Add experience in replay store memory"""

        # Put data inside arrays
        self.__state[self.__data_pointer] = state
        self.__action[self.__data_pointer] = action
        self.__reward[self.__data_pointer] = reward
        # self.__next_state[self.__data_pointer] = next_state
        self.__done[self.__data_pointer] = done

        # Update prioritized tree. Obs: Fill the leaves from left to right
        tree_index = self.__data_pointer + self.__capacity - 1
        self.update(tree_index, priority)

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

    def total_priority(self):
        """Get total priority's tree"""
        return self.__tree[0]

    def get_priotiry(self):
        return self.__tree[-self.__capacity:]


spec_memory = [
    ("__per_e", float64),
    ("__per_a", float64),
    ("__per_b", float64),
    ("__per_b_increment_per_sampling", float64),
    ("__absolute_error_uper", float64),
    ("__tree", SumTree),
]


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class Memory:
    def __init__(self, size_board, capacity):
        self.__capacity = capacity
        # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
        self.__per_e = 0.01

        # Hyperparameter used to make a tradeoff between taking only high priority and sampling randomly
        self.__per_a = 0.6

        # Importance-sampling, from initial value increasing to 1
        self.__per_b = 0.4

        # Increment per_b per sampling step
        self.__per_b_increment_per_sampling = 0.001

        # Clipped abs error
        self.__absolute_error_upper = 1.

        self.__tree = SumTree(capacity, size_board)

    spec_store = [("max_priority", float64)]

    def store(self, state, action, reward, next_state, done):
        # Find the maximum priotiry
        max_priority = np.max(self.__tree.get_priotiry())
        # max_priority = 0

        # We can't put priority = 0 since this exp will never being taken
        if max_priority == 0:
            max_priority = self.__absolute_error_upper

        # add exprience in tree
        self.__tree.add(max_priority, state, action, reward, next_state, done)

    def sample(self, batch_size):
        # Create a sample array that will contains the minibatch
        memory_batch = []

        batch_idx, batch_ISWeights = (
            np.empty((batch_size,), dtype=np.int32),
            np.empty((batch_size, 1), dtype=np.float32),
        )

        # Calculate the priority segment
        priority_segment = self.__tree.total_priority() / batch_size

        # Increasing per_b by per_b_increment_per_sampling
        self.__per_b = np.min(
            [1., self.__per_b + self.__per_b_increment_per_sampling]
        )  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.__tree.get_priotiry()) / self.__tree.total_priority()
        max_weight = (p_min * batch_size) ** (-self.__per_b)

        for i in range(batch_size):
            # A value is uniformly sample from each range
            limit_a, limit_b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(limit_a, limit_b)

            # Experience that correspond to each value is retrieved
            index, priority, state, action, reward, next_state, done = self.__tree.get_leaf(
                value
            )

            # P(j)
            sampling_probabilities = priority / self.__tree.total_priority()

            #  IS = (1/batch_size * 1/P(i))**per_b /max wi == (Batch_size*P(i))**-per_b  /MAX(weight)
            batch_ISWeights[i, 0] = (
                np.power(batch_size * sampling_probabilities, -self.__per_b)
                / max_weight
            )

            batch_idx[i] = index
            memory_batch.append(Transition(state, action, reward, next_state, done))

        return batch_idx, memory_batch, batch_ISWeights

    def batch_update(self, tree_indexes, abs_errors):
        """Update priorities on the tree"""
        abs_errors += self.__per_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.__absolute_error_upper)
        priorities = np.power(clipped_errors, self.__per_a)

        for tree_index, priority in zip(tree_indexes, priorities):
            self.__tree.update(tree_index, priority)
