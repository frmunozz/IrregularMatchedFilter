import numpy as np

_ALLOWED_DTYPES = [np.float32, np.float64, np.complex64,
                   np.complex128, np.uint32, np.int32, np.int64]

_ALLOWED_SCALARS = [int, float, complex] + _ALLOWED_DTYPES


def zeros(length, dtype=None):
    """
    Return an Array type filled with zeros

    :param length: lenghth of the input array to Array type
    :param dtype: dtype of the Array
    :return: Array object
    """
    if dtype is None:
        dtype = np.float64

    return Array(np.zeros(length), dtype=dtype)


class Array(object):
    def __init__(self, initial_array, dtype=None):
        """
        Array used to do numeric calculations in order to simplify posteriori
        syntax. It is a convenient wrapped around numpy. Original idea should
        be use methods declared in PyCBC base code but since it's under
        developing for python 3.x we need to write the classes.

        :param initial_array:   array-like object (list, numpy.array)
                                data type to be wrapped in Array class
        :param dtype:           numpy style dtype
                                type of the encapsulated data
        """
        if isinstance(initial_array, Array):
            initial_array = initial_array.value
        else:
            initial_array = np.array(initial_array)

        if dtype is not None:
            dtype = np.dtype(dtype)
        elif initial_array.dtype in _ALLOWED_DTYPES:
            dtype = initial_array.dtype
        else:
            raise TypeError('input array data type'
                            ' %d not allowed' % initial_array.dtype)

        if initial_array.dtype != dtype:
            initial_array = initial_array.astype(dtype)

        self._data = initial_array

    def __len__(self):
        """
        Return length of Array

        :return: integer
        """
        return len(self._data)

    def __abs__(self):
        """
        Return absolute value of Array

        :return: Array
        """
        return abs(self._data)

    def __str__(self):
        """
        Return Array casted into string.

        :return: string
        """
        return str(self._data)

    def __mul__(self, other):
        """
        Multiply by an Array or a scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data * other

    __rmul__ = __mul__

    def __imul__(self, other):
        """
        Multiply by an Array or a scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        self._data *= other
        return self

    def __add__(self, other):
        """
        Add Array to Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data + other

    __radd__ = __add__

    def __iadd__(self, other):
        """
        Add Array to Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        self._data += other
        return self

    def __floordiv__(self, other):
        """
        Integer Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data // other

    def __rfloordiv__(self, other):
        """
        Integer Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data.__rfloordiv__(other)

    def __ifloordiv__(self, other):
        """
        Integer Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        self._data //= other
        return self

    def __truediv__(self, other):
        """
        Float Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data / other

    def __rtruediv__(self, other):
        """
        Float Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        return self._data.__rtruediv__(other)

    def __itruediv__(self, other):
        """
        Float Divide Array by Array or scalar and return an Array.

        :param other: Array or scalar
        :return: Array
        """
        self._data /= other
        return self

    def __sub__(self, other):
        """
        Subtract Array or scalar from Array and return an Array.

        :param other: Array or scalar (int, float, complex, etc)
        :return: Array
        """
        return self._data - other

    def __rsub__(self, other):
        """
        Subtract Array or scalar from Array and return an Array.

        :param other: Array or scalar (int, float, complex, etc)
        :return: Array
        """
        return self._data.__rsub__(other)

    def __isub__(self, other):
        """
        Subtract Array or scalar from Array and return an Array.

        :param other: Array or scalar (int, float, complex, etc)
        :return: Array
        """
        self._data -= other
        return self

    def __eq__(self, other):
        """
        Python magic method called when the '==' comparison is used.
        We define this to return true when the data of two Array are identical
        and all the meta data are identical too.

        :param other:
        :return:
        """
        if type(self) != type(other):
            return False
        if self.dtype != other.dtype:
            return False
        if len(self) != len(other):
            return False

        return (self.value == other.value).all()

    def __getitem__(self, index):
        """
        Return items from the Array. This not guaranteed to be fast for
        returning single values.

        :param index: slice instance of indexes or list
        :return:
        """
        if isinstance(index, list):
            index = slice(index[0], index[1])

        if isinstance(index, slice):
            return self._getslice(index)
        else:
            return self._data[index]

    def __setitem__(self, index, other):
        """
        Set an slice of the Array to a given value

        :param index: slice instance of indexes or list of length 2
        :param other: Array instance or array-like (list, numpy.array)
        """
        if isinstance(index, tuple):
            index = slice(index[0], index[1])

        if isinstance(index, int):
            index = slice(index, index + 1)

        if isinstance(other, Array):
            if self.kind is 'real' and other.kind is 'complex':
                raise ValueError('Cannot set real value with complex')

            if isinstance(index, slice):
                self._data[index] = self._copy(self._data[index], other._data)
            else:
                raise ValueError("index must be a list of two elements "
                                 "or slice")

        elif type(other) in _ALLOWED_SCALARS:
            if isinstance(index, slice):
                self[index].fill(other)
            else:
                raise ValueError("index must be a list of two elements "
                                 "or slice")

    @property
    def value(self):
        return self._data

    @property
    def real(self):
        return self._data.real

    @property
    def imag(self):
        return self._data.imag

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def ndim(self):
        return self._data.ndim

    @property
    def kind(self):
        if self.dtype == np.float32 or self.dtype == np.float64:
            return 'real'
        elif self.dtype == np.int32 or self.dtype == np.int64:
            return 'real'
        elif self.dtype == np.complex64 or self.dtype == np.complex128:
            return 'complex'
        else:
            return 'unknown'

    def _return(self, ary, **kwargs):
        """
        Wrap the ary to return an Array type

        :param ary:
        :return:
        """
        if isinstance(ary, Array):
            return ary

        return Array(ary)

    def _getslice(self, index):
        return self._return(self._data[index])

    def _copy(self, self_ary, other_ary):
        if len(self_ary) <= len(other_ary):
            idx = slice(0, len(self_ary))
            self_ary = other_ary[idx]
        else:
            idx = slice(0, len(other_ary))
            self_ary[idx] = other_ary
        return self_ary

    def conj(self):
        return self._data.conj()

    def norm(self, weight=None):
        """
        Return the element wise squared norm of the array

        :return:
        """
        return np.sqrt(self.weighted_inner(weight=weight))

    def squared_norm(self):
        """ Return the elementwise squared norm of the array """
        return self.real ** 2 + self.imag ** 2

    def inner(self):
        """
        Return the inner product of the array with complex conjugation.

        :return:
        """
        # return self.vdot(self)
        return (self._data * self.conj()).sum()

    def weighted_inner(self, weight=None):
        """
        Return the inner product of the array with complex conjugation and
        wheighted by 'weight' Array or scalar value

        :param weight: Array or scalar
        :return:
        """
        if weight is None:
            weight = 1

        # return self.vdot(self / weight)
        return (self._data * self.conj() / weight).sum()

    def sum(self):
        """
        Return the sum of the the array.

        :return:
        """
        return self._data.sum()

    def max(self):
        """
        Return the maximum value in the array. Only support real arrays

        :return:
        """
        if self.kind != "real":
            raise TypeError(self.__class__.__name__ +
                            " does not support complex types")
        return self._data.max()

    def argmax(self):
        """
        Return the index of the maximum value in the array. Only support real
        arrays

        :return:
        """
        if self.kind != "real":
            raise TypeError(self.__class__.__name__ +
                            " does not support complex types")
        return self._data.argmax()

    def min(self):
        """
        Return the minimum value in the array. Only support real arrays

        :return:
        """
        if self.kind != "real":
            raise TypeError(self.__class__.__name__ +
                            " does not support complex types")
        return self._data.min()

    def argmin(self):
        """
        Return the index of the minimum value in the array. Only support real
        arrays

        :return:
        """
        if self.kind != "real":
            raise TypeError(self.__class__.__name__ +
                            " does not support complex types")
        return self._data.argmin()

    def resize(self, new_size):
        """
        resize array to new_size

        :param new_size:
        :return:
        """
        if new_size == len(self):
            return
        else:
            new_arr = np.zeros(new_size, dtype=self.dtype)
            if len(self) < new_size:
                new_arr[0:len(self)] = self
            else:
                new_arr = self[0:new_size]

            self._data = new_arr

    def replace(self, new_data):
        """
        replace the array values by new_data

        :param new_data:
        """
        self._data = new_data

    def roll(self, shift):
        """
        roll the array value by shift indexes

        :param shift:
        """
        self._data = np.roll(self._data, shift)
        return self

    def fill(self, scalar):
        self._data = (np.ones(len(self)) * scalar).astype(self.dtype)

    def dot(self, other):
        """
        Return the dot product between self Array and other Array.
        This product doesn't handle complex value

        :param other: Array type or array-like
        """
        if isinstance(other, Array):
            return np.dot(self.value, other.value)
        else:
            return np.dot(self.value, other)

    def vdot(self, other):
        """
        Return the dot product between self Array and other Array.
        This product handle complex value, taking the complex conjugation of
        self.

        :param other:
        :return:
        """
        if isinstance(other, Array):
            return np.vdot(self.value, other.value)
        else:
            return np.vdot(self.value, other)

    def delete(self, idxs:slice):
        """
        delete a part from the array
        :param idxs: slice of the array to delete
        """
        self._data = np.delete(self._data, idxs)

    def windowed(self, window: np.ndarray, **kwargs):
        """
        window the data values and return a copy of the array

        :param window: the window array
        """
        return self._return(self._data * window, **kwargs)

    def abs_max_loc(self):
        new_self = abs(self)
        return new_self.max(), new_self.argmax()

    def copy(self):
        copy_data = np.copy(self._data)
        return self._return(copy_data)

    def add_point(self, point):
        self._data = np.append(self._data, point)

