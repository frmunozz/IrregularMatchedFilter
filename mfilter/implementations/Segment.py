import numpy as np
import pdb

class DataSegment:
    def __init__(self, t, y, t_segment=None, overlap_factor=None):
        self.t, self.y = self._validate_data(t, y)
        self.t_duration = max(t) - min(t)
        self.stride, self.t_segment, self.n_segment = self._prm(overlap_factor,
                                                                t_segment)
        self.counter_segments = 0

    def _validate_data(self, t, y):
        if t.shape != y.shape:
            raise ValueError("Inputs (t, y) must be 1-dimensional and " +
                             "same shape")
        return t, y

    def _prm(self, overlap_factor, t_segment):
        if overlap_factor is None:
            # by default ise overlap of 50%, this factor is
            # the part that is not overlapped
            overlap_factor = 0.5

        stride = t_segment * overlap_factor  # overlap in time

        if t_segment is None:
            t_segment = self.t_duration
            n_segments = 1
        else:
            if t_segment > self.t_duration:
                raise ValueError("input t_segment cannot be higher than "
                                 "duration of times (max(t) - min(t))")
            elif t_segment == self.t_duration:
                n_segments = 1

            else:
                division = self.t_duration / t_segment
                n_segments = int(np.ceil(division / overlap_factor)) - 1

        return stride, t_segment, n_segments

    def check_segment_number(self, segment_number):

        if segment_number + 1 > self.n_segment:
            raise ValueError("input segment_number cannot be higher "
                             "than {}".format(self.n_segment-1))
        if segment_number < 0:
            raise ValueError("input segment_number cannot be less than 0")

    def _end_time(self, start_at_time):
        end_at_time = start_at_time + self.t_segment
        difference = self.t_duration - end_at_time
        if difference < 0:
            return start_at_time + difference, end_at_time + difference
        else:
            return start_at_time, end_at_time

    def compute_segment(self, segment_number: int = None):
        # pdb.set_trace()
        self.check_segment_number(segment_number)
        start_at_time = segment_number * self.stride
        start_at_time, end_at_time = self._end_time(start_at_time)
        idx_start = np.argmin(np.abs(self.t - start_at_time + min(self.t)))
        idx_end = np.argmin(np.abs(self.t - end_at_time + min(self.t)))
        return self.t[idx_start:idx_end+1], self.y[idx_start:idx_end+1]

    def set_segment_number(self, segment_n=None):
        if segment_n is None:
            segment_n = self.counter_segments
            self.counter_segments += 1

        if segment_n >= self.n_segment:
            raise ValueError("cannot compute data segment for " +
                             "segment {} because the ".format(segment_n + 1) +
                             "maximum segment allowed " +
                             "is {}".format(self.n_segment))
        return segment_n


