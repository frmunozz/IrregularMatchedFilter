import numpy as np
from scipy import signal


class Templates:
    def __init__(self, template_type=None, prm: dict=None):
        self.prm = prm
        self.template_type = self._validate_template(template_type, prm)

    def _validate_template(self, template_type, prm):

        if template_type is None:
            template_type = 'one_sin_default'
            self.prm = {'default': True}

        elif 'sin' in template_type:
            if any([x not in prm for x in ['freq', 'weight']]):
                raise ValueError("for template type 'one_sin', kwargs" +
                                 "must contain keys " +
                                 "{'freq', 'weight'}")
            self.prm['default'] = False

        elif template_type == 'gaussian':
            if any([x not in prm for x in ['std', 'weight']]):
                raise ValueError("for template type 'one_sin', kwargs" +
                                 "must contain keys " +
                                 "{'std','weight'}")

        else:
            raise NotImplementedError("the currently implemented " +
                                      "options for template_type are " +
                                      "{'one_sin', 'gaussian'}")

        return template_type

    def normalize_times(self, times):
        return times - min(times)

    def _sin_template(self, times, default=True):
        temp = np.zeros(len(times))
        if default:
            freq = 0.5 / max(times)
            w = 1
            self.prm['freq'] = freq
            self.prm['weight'] = w
            self.prm['default'] = False
        else:
            freq = self.prm['freq']
            w = self.prm['weight']

        idx_max = np.argmin(np.abs(times - 0.5 / freq))
        temp[0:idx_max] = w * np.sin(freq * 2 * np.pi * times[0:idx_max])
        return temp

    def get_period(self):
        if 'freq' not in self.prm:
            raise ValueError("cannot get frequency from " +
                             format(self.template_type) +
                             " template")
        return 0.5 / self.prm['freq']

    def _gauss_template(self, times):
        w = self.prm['weight']
        return w * signal.gaussian(len(times), self.prm['std'])

    def compute_template(self, times):
        times = self.normalize_times(times)
        if self.template_type == 'gaussian':
            return self._gauss_template(times)
        else:
            return self._sin_template(times, default=self.prm['default'])



