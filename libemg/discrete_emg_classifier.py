from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import time
from libemg.utils import get_windows

class DiscreteEMGClassifier:
    """The Offline Discrete EMG Classifier. 

    This is the base class for any offline EMG classification for dynamic gestures (many windows to one output). 

    Parameters
    ----------

    """
    def __init__(self):
        self.gesture_recognizer = None
        self.blip_detector = None 

    def fit(self, gr, bd):
        self.gesture_recognizer = gr 
        self.blip_detector = bd 
 
class OnlineEMGDiscreteClassifier:
    """OnlineEMGClassifier.

    Given a DiscreteEMGClassifier and additional information, this class will stream class predictions over UDP in real-time.

    Parameters
    ----------
    offline_classifier: EMGClassifier
        An EMGClassifier object. 
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    """
    def __init__(self, offline_classifier, online_data_handler, port=12346, ip='127.0.0.1', modalities={}):
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier
        self.modalities = modalities # Parameters: name, window_size, window_increment, features, mean, std 
        self.raw_data = online_data_handler.raw_data
        self.process = Process(target=self._run_helper, daemon=True,)
        self.feats = []
        
    def run(self, block=True):
        """Runs the classifier - continuously streams predictions over UDP.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        if block:
            self._run_helper()
        else:
            self.process.start()

    def stop_running(self):
        """Kills the process streaming classification decisions.
        """
        self.process.terminate()

    def _run_helper(self):
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        while True:
            # Lets base everything off of EMG 
            if len(self.raw_data.get_others()['IMU-bio']) >=90 and len(self.raw_data.get_others()['EMG-bio']) >= 3600: # and len(self.raw_data.get_others()['PPG-bio']) >= 90:

                emg_data = np.array([self.raw_data.get_others()['EMG-bio'][-3600:]])
                emg_data = self.normalize(emg_data, self.modalities['EMG']['mean'], self.modalities['EMG']['std'])

                imu_data = np.array([np.array(self.raw_data.get_others()['IMU-bio'])[-90:, 0:3]])
                imu_data = self.normalize(imu_data, self.modalities['IMU']['mean'], self.modalities['IMU']['std'])

                ppg_data = np.array([self.raw_data.get_others()['PPG-bio'][-90:]])
                ppg_data = self.normalize(ppg_data, self.modalities['PPG']['mean'], self.modalities['PPG']['std'])

                self.raw_data.adjust_increment_other('EMG-bio', 3600, 200)
                self.raw_data.adjust_increment_other('IMU-bio', 90, 5)
                self.raw_data.adjust_increment_other('PPG-bio', 90, 5)

                emg_feats = self.get_features(fe, emg_data, 400, 100, ['RMS'])
                imu_feats = self.get_features(fe, imu_data, 10, 5, ['MEAN'])
                ppg_feats = self.get_features(fe, ppg_data, 10, 5, ['RMS'])

                preds = self.classifier.blip_detector.predict(imu=imu_feats) #, imu=imu_feats, ppg=ppg_feats)
                print(preds)

    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
    
    def normalize(self, data, mean, std):
        if data.shape[2] == 1:
            return (data - mean)/std
        for ch in range(data.shape[2]):
            if std[ch] != 0:
                data[:,:,ch] = (data[:,:,ch] - mean[ch]) / std[ch]
        return data
    
    def get_features(self, fe, data, window_size, window_inc, feats):
        data = np.array([get_windows(d, window_size, window_inc) for d in data])
        feats = np.array([fe.extract_features(feats, d, array=True) for d in data])
        feats = np.nan_to_num(feats, copy=True, nan=0, posinf=0, neginf=0)
        return feats 
    