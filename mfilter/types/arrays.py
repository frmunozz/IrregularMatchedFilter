import numpy as np


class Array(object):
    def __init__(self, initial_array, delta=-1):
        initial_array = np.array(initial_array)

        self._delta = delta
        self.offset = min(initial_array)
        self.end = max(initial_array)
        self._data = initial_array

    def __len__(self):
        return len(self._data)

    def __abs__(self):
        return abs(self._data)

    def __str__(self):
        return str(self._data)

    def __mul__(self, other):
        return self._data * other

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self) != len(other):
            return False
        if self.delta != other.delta:
            return False
        if self.duration != other.duration:
            return False
        return True

    @property
    def duration(self):
        return self.end - self.offset

    @property
    def delta(self):
        return self._valid_delta()

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    def _average_delta(self):
        return (self.end - self.offset) / len(self)

    def _valid_delta(self):
        if self._delta <= 0:
            return self._average_delta()
        else:
            return self._delta

    def _reset(self):
        self._delta = self.delta
        self.offset = min(self._data)
        self.end = max(self._data)

    def resize(self, new_size):

        if new_size == len(self):
            return
        else:
            new_arr = np.zeros(new_size, dtype=self.dtype)
            if len(self) < new_size:
                new_arr[0:len(self)] = self
            else:
                new_arr = self[0:new_size]

            self._data = new_arr
            self._reset()

    def replace(self, new_data):
        self._data = new_data

    def roll(self, shift):
        self._data = np.roll(self._data, shift)

    def slice_by_values(self, start, end):
        if start < self.offset:
            raise ValueError(
                'start point given is less than the minimum value in Array')

        if end > self.end:
            raise ValueError(
                'end point given is higher than the maximu value in Array')

        start_idx = np.argmin(np.abs(self.offset - start))
        end_idx = np.argmin(np.abs(self.end - end))
        return self.slice_by_indexes(start_idx, end_idx)

    def slice_by_indexes(self, start_idx, end_idx):
        return self._data[start_idx:end_idx+1]

    def set_data(self, **kwargs):
        pass

    def dot(self, other):
        pass

