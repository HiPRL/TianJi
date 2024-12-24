# -*- coding: utf-8 -*-
import copy
import functools
import random
from collections import deque, namedtuple
from threading import Lock

__all__ = ["Experience", "ReplayMemory", "buffer_lock"]


global_lock = Lock()
Experience = namedtuple(
    "Experience",
    ["index", "state", "action", "reward", "next_state", "terminal", "other_args"],
)


def buffer_lock(func):
    func.__lock__ = global_lock

    @functools.wraps(func)
    def wraper(*args, **kwargs):
        with func.__lock__:
            return func(*args, **kwargs)

    return wraper


class ReplayMemory(object):
    def __init__(self, max_size):
        assert isinstance(
            max_size, int
        ), f"TypeError, max_size should be int, but got {type(max_size)}"
        self.next_idx = 0
        self.max_size = max_size
        self.pool = deque([], maxlen=max_size)

    def __len__(self):
        return len(self.pool)

    def __iter__(self):
        return copy.deepcopy(self)

    def __next__(self):
        if len(self.pool) > 0:
            return self.pool.popleft()
        else:
            raise StopIteration()

    @buffer_lock
    def __getitem__(self, key):
        return self.pool[key]

    @buffer_lock
    def __setitem__(self, key, value):
        self.pool[key] = value

    @buffer_lock
    def push(self, *args, force=False):
        if force:
            self.pool.extend(*args)
            self.next_idx = (self.next_idx + len(*args)) % self.max_size
        else:
            assert (
                len(args) >= 6
            ), f"The number of parameters is at least greater than 6, and the order is 'index' 'state' 'action' 'reward' 'next_state' 'terminal' 'other_args'"
            self.pool.append(Experience(*args))
            self.next_idx = (self.next_idx + 1) % self.max_size

    @buffer_lock
    def pop(self, batch_size):
        assert (
            len(self.pool) >= batch_size
        ), f"The buffer data is insufficient for pop, only {len(self.pool)} data but want get {batch_size} data."
        return [next(self) for _ in range(batch_size)]

    @buffer_lock
    def clear(self):
        while len(self.pool):
            self.pool.pop()

    @buffer_lock
    def _sample_batch(self, batch_size):
        assert (
            batch_size <= self.size
        ), f"batch_size out of range, pool max size {self.size}!"
        return random.sample(self.pool, batch_size)

    @buffer_lock
    def _pluck_sample(self, idxes):
        return [self.pool[i] for i in idxes]

    @property
    def is_overflow(self):
        return self.max_size == len(self.pool)

    @property
    def index(self):
        return self.next_idx

    @property
    def size(self):
        return self.max_size
