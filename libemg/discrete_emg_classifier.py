from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import pickle
import socket
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import inspect
from scipy import stats

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
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: list or None
        A list of features that will be extracted during real-time classification. These should be the 
        same list used to train the model. Pass in None if using the raw data (this is primarily for CNNs).
    parameters: dict (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    tcp: bool (optional), default = False
        If True, will stream predictions over TCP instead of UDP. 
    """
    def __init__(self, offline_classifier, window_size, window_increment, gesture_len, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False):
        self.window_size = window_size
        self.window_increment = window_increment
        self.gesture_len = gesture_len
        self.raw_data = online_data_handler.raw_data
        self.filters = online_data_handler.fi
        self.features = features
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier
        self.window_buffer = []

        self.tcp = tcp
        if not tcp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            print("Waiting for TCP connection...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.bind((ip, port))
            self.sock.listen()
            self.conn, addr = self.sock.accept()
            print(f"Connected by {addr}")

        self.process = Process(target=self._run_helper, daemon=True,)
        self.std_out = std_out
        
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
            if len(self.raw_data.get_emg()) >= self.window_size:
                data = self._get_data_helper()

                # Extract window 
                window = get_windows(data[-self.window_size:][:], self.window_size, self.window_size)
                features = fe.extract_features(self.features, window, self.classifier.feature_params)
                # If extracted features has an error - give error message
                if (fe.check_features(features) != 0):
                    self.raw_data.adjust_increment(self.window_size, self.window_increment)
                    continue
                classifier_input = self._format_data_sample(features)
                self.window_buffer.append(classifier_input)
                self.raw_data.adjust_increment(self.window_size, self.window_increment)

                # With window buffer, make prediction 
                self.window_buffer = self.window_buffer[-100:] # Keep buffering this so lag doesn't build up

                if self.classifier.blip_detector.predict(self.window_buffer[-self.gesture_len:]) #TODO: Update this 
                    # See if it is the correct gesture 
                    if self.classifier.gesture_recognizer.predict(self.window_buffer[-self.gesture_len:]):
                        print("Wake Gesture Recognized!")

    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr

    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data
    
    