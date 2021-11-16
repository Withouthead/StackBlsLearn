import pandas as pd
import numpy as np
import os
import time

class StackBlsLogger:
    def __init__(self, save_path, file_name):
        self.build_time = time.localtime(time.time())
        self.save_path = os.path.join(save_path, time.strftime("%Y_%m_%d.csv", self.build_time))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.file_name = file_name
        self.file_path = os.path.join(save_path, self.file_name)
        self.result = None

    def log(self, log_value):
        if not isinstance(log_value, np.ndarray):
            log_value = np.array(log_value)
        log_value.reshape(1, -1)
        if self.result is None:
            self.result = log_value
        else:
            self.result = np.concatenate(self.result, log_value, 0)

    def save(self):
        np.savetxt(self.file_path, self.result, delimiter=',')