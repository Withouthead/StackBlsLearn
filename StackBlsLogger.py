import pandas as pd
import numpy as np
import os
import time
import json
class StackBlsLogger:
    def __init__(self, save_path, file_name, data_set_name, config):
        self.build_time = time.localtime(time.time())
        self.save_path = os.path.join(save_path, time.strftime("%Y_%m_%d", self.build_time))


        self.file_name = file_name
        self.file_path = os.path.join(self.save_path)
        self.result = None
        self.config = {
            "dataset":data_set_name
        }
        for i in range(len(config)):
            self.config["stack" + str(i)] = {
                "feature_size" : config[i][0],
                "windows_size": config[i][1],
                "init_enhance_size": config[i][2],
                "increase_size" : config[i][3],
                "init_increase_size" : config[i][4],
                "increase_step_size": config[i][5]
            }

        self.config = json.dumps(self.config, indent=4)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)


    def log(self, log_value):
        if not isinstance(log_value, np.ndarray):
            log_value = np.array(log_value)
        log_value = np.expand_dims(log_value, axis=0)
        if self.result is None:
            self.result = log_value
        else:
            self.result = np.concatenate((self.result, log_value), 0)

    def save(self):
        np.savetxt(os.path.join(self.file_path, self.file_name), self.result, delimiter=',')
        config_file = open(os.path.join(self.file_path, "config.json"), "w")
        config_file.write(self.config)
        config_file.close()