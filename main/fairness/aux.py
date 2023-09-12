from scipy.interpolate import interp1d
import numpy as np

class EQF:
    def __init__(self, 
                 sample_data: np.ndarray,
                 ):
        self._calculate_eqf(sample_data)

    def _calculate_eqf(self,sample_data):
        sorted_data = np.sort(sample_data)
        linspace  = np.linspace(0,1,num=len(sample_data))
        self.interpolater = interp1d(linspace, sorted_data)
        self.min_val = sorted_data[0]
        self.max_val = sorted_data[-1]

    def __call__(self, value_):
        try:
            return self.interpolater(value_)
        except ValueError:
            if value_ < self.min_val:
                return 0.0
            elif value_ > self.max_val:
                return 1.0
            else:
                raise ValueError('Error with input value')