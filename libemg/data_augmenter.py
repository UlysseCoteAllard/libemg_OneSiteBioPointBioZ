import numpy as np
import random
from scipy import interpolate
from sktime.transformations.panel.interpolate import TSInterpolator

class DataAugmenter:
    """
    Data augmentation class.
    """
    def get_augmentation_list(self):
        """Gets a list of all available augmentations.
        
        Returns
        ----------
        list
            A list of all available augmentations.
        """
        return ['GNOISE',       # Add gaussian noise 
                'CS',           # Channel swapping
                'CR',           # Channel rotation 
                'CROP',         # Randomly crop out part of signal 
                'MAG',          # Scale the signal by some factor 
                'WARP',         # Stretch/shrink the signal by some factor 
                'FLIP',         # Flips the signal 
                'POL',          # Flips the polarity of the signal
                'ZERO',         # Randomly zeros values
                'MIXUP',        # Randomly mixes up the array
                ]  

    def augMIXUP(self, data, percent=0.05, mixups=5):
        for _ in range(0, 5):
            start_idx = random.randint(1, len(data) - len(data) * percent)
            data = np.vstack([data[0:start_idx,:], data[int(start_idx+len(data)*percent):len(data)], data[start_idx:int(start_idx+len(data)*percent):,:] ])
        return data 

    def augWARP(self, data, min=0.8, max=1.2):
        warp_amt = int(random.uniform(min, max) * len(data))
        interp = TSInterpolator(warp_amt)
        return interp.fit_transform(data)

    def augCROP(self, data, percent=0.05):
        start_idx = random.randint(1, len(data) - len(data) * percent)
        return np.vstack([ data[0:start_idx,:], data[int(start_idx+len(data)*percent):len(data)]])

    def augZERO(self, data, percent=0.05):
        vals = np.random.choice([0, 1], size=(len(data),), p=[percent, 1-percent])
        return (data.T * vals).T

    def augGNOISE(self, data, mag=1):
        gaussian_noise = np.random.normal(np.mean(data, axis=0) * mag, np.std(data, axis=0) * mag, data.shape)
        return data + gaussian_noise

    def augPOL(self, data):
        return data * -1

    def augMAG(self, data, min=0, max=5):
        return data * random.uniform(min, max)
        
    def augCS(self, data):
        return data[:, np.random.permutation(data.shape[1])]
    
    def augCR(self, data):
        n_data = []
        if np.random.randint(0,2) == 1:
            # Swap Right 
            for i in range(0, len(data[0])):
                if i != len(data[0]) - 1:
                    n_data.append(data[:,i+1])
                else:
                    n_data.append(data[:, 0])
        else:
            # Swap Left
            for i in range(0, len(data[0])):
                if i != 0:
                    n_data.append(data[:,i-1])
                else:
                    n_data.append(data[:, -1])
        return np.transpose(n_data)

    def augFLIP(self, data):
        return np.flip(data, axis=0)