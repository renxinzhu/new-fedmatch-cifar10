from typing import Union, Dict
from datetime import datetime
import os
import torch

#should be moved to hyper_parameters.py later
LOG_PATH = "./log"
MODEL_PATH = "./model"

class Logger:
    def __init__(self):
        path = './log'
        filename = f'fed-match - {datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt'

        if not os.path.isdir(path):
            os.makedirs(path)

        self.file_path = os.path.join(path, filename)

    def log(self, content: str):
        print(content)

        with open(self.file_path, 'a+') as outfile:
            outfile.write(content)
            outfile.write('\n')

class Logger:
    def __init__(self, client_id: Union[int, None], device = torch.device):
        self.client_id = client_id
        self.filepath = None
        self.log_path = LOG_PATH
        self.device = device
        #model path
        self.name = f'client-{self.client_id}' if self.client_id else 'server'
        self.model_path = os.path.join(MODEL_PATH, f'{self.name}.pt')

    def print(self, message):
        print(f'[{datetime.now().strftime("%Y/%m/%d-%H:%M:%S")}]' +
              f'[{self.name}] ' +
              f'{message}')

    def save_current_state(self, data):
        filename = f'{self.name}-{datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}.txt'
        if not self.filepath:
            self.filepath = os.path.join(self.log_path, filename)

        if not os.path.isdir(self.log_path):
            os.makedirs(self.log_path)

        with open(self.filepath, 'a+') as outfile:
            content = [str(i) for i in data.values()]
            outfile.write(",".join(content))
            outfile.write('\n')
