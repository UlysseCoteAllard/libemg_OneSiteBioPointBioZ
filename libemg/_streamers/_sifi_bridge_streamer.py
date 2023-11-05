import time
import os
import requests
# this is responsible for receiving the data
class SiFiBridge:
    def __init__(self, config, version):
        self.version = version
        import platform
        # note, for linux you may need to use sudo chmod +x sifi_bridge_linux
        if platform.system() == 'Linux':
            if not os.path.exists('sifi_bridge_linux'):
                r = requests.get('https://raw.githubusercontent.com/eeddy/libemg/sifi-bioarmband/libemg/_streamers/sifi_bridge_linux?token=GHSAT0AAAAAACDJYY4C56HYEGNCFDJGJUPGZFPYBKA')
            with open("sifi_bridge_linux", "wb") as file:
                    file.write(r.content)
                    print("Please run <sudo chmod +x sifi_bridge_linux> in the terminal to indicate this is an executable file! You only need to do this once.")
            self.proc = subprocess.Popen(['sifi_bridge_linux'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        elif platform.system() == "Windows":
            if not os.path.exists('sifi_bridge_windows.exe'):
                r = requests.get('https://raw.githubusercontent.com/eeddy/libemg/sifi-bioarmband/libemg/_streamers/sifi_bridge_windows.exe?token=GHSAT0AAAAAACDJYY4CXIP2ELDFQOWK6WLEZFP5SWA')
                with open("sifi_bridge_windows.exe", "wb") as file:
                    file.write(r.content)
            self.proc = subprocess.Popen(['sifi_bridge_windows.exe'],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE)
        self.config = config
        self.emg_handlers = []

    def connect(self):
        connected = False
        while not connected:
            name = '-c BioPoint_v' + str(self.version) + '\n'
            self.proc.stdin.write(bytes(name, 'utf-8'))
            self.proc.stdin.flush()

            ret = self.proc.stdout.readline().decode()

            dat = json.loads(ret)

            if dat["connected"] == 1:
                connected = True
            else:
                print("Could not connect. Retrying.")
        # Setup channels
        self.proc.stdin.write(self.config)
        self.proc.stdin.flush()


    def add_emg_handler(self, closure):
        self.emg_handlers.append(closure)

    def run(self):
        self.proc.stdin.write(b'-cmd 1\n')
        self.proc.stdin.flush()
        self.proc.stdin.write(b'-cmd 0\n')
        self.proc.stdin.flush()
        packet = np.zeros((14,8))
        while True:
            data_arr_as_json = self.proc.stdout.readline().decode()
            if data_arr_as_json == "" or data_arr_as_json.startswith("sending cmd"):
                continue
            data_arr_as_json = json.loads(data_arr_as_json)
            if "data" in list(data_arr_as_json.keys()):
                if "emg0" in list(data_arr_as_json["data"].keys()):
                    for c in range(packet.shape[1]):
                        packet[:,c] = data_arr_as_json['data']["emg"+str(c)]
                    for s in range(packet.shape[0]):
                        for h in self.emg_handlers:
                            h(packet[s,:].tolist())
                elif "emg" in list(data_arr_as_json["data"].keys()): # This is the biopoint emg 
                    emg = data_arr_as_json['data']["emg"]
                    for e in emg:
                        self.emg_handlers[0]([e])

    def close(self):
        self.proc.stdin.write(b'-cmd 1\n')
        self.proc.stdin.flush()
        return

    def turnoff(self):
        self.proc.stdin.write(b'-cmd 13\n')
        self.proc.stdin.flush()
        return

import subprocess
import json
import numpy as np
import socket
import pickle
class SiFiBridgeStreamer:
    def __init__(self, ip, port, version='1_2',
                 ecg=False,
                 emg=True, 
                 eda=False,
                 imu=False,
                 ppg=False,
                 notch_on=True,
                 notch_freq = 60,
                 emgfir_on=True,
                 emg_fir = [20, 450]):
        # notch_on refers to EMG notch filter
        # notch_freq refers to frequency cutoff of notch filter
        # 
        self.version = version
        self.ip = ip 
        self.port = port
        self.config = "-s ch " +  str(int(ecg)) +","+str(int(emg))+","+str(int(eda))+","+str(int(imu))+","+str(int(ppg)) + " " 
        if notch_on or emgfir_on:
            self.config += " enable_filters 1 "
            if notch_on:
                self.config += " emg_notch " + str(notch_freq)
            else:
                self.config += " emg_notch 0"
            if emgfir_on:
                # NOTE: notch flag should be on to do bandpass stuff
                self.config += " emg_fir " + str(emg_fir[0]) + "," + str(emg_fir[1]) + ""


        else:
            self.config += " enable_filters 0 "

        print(self.config)
        self.config = bytes(self.config,"UTF-8")

    def start_stream(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        b= SiFiBridge(self.config, self.version)
        b.connect()

        def write_emg(emg):
            data_arr = pickle.dumps(list(emg))
            sock.sendto(data_arr, (self.ip, self.port))
        b.add_emg_handler(write_emg)

        while True:
            try:
                b.run()
            except Exception as e:
                print("Error Occured: " + str(e))
                continue
                