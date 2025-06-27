import time
import os
import torch
import numpy as np
 
flag = os.getenv('MEASURE_BREAKDOWN', False)
if flag:
    print("BREAKDOWN_FLAG setted up!")
 
class Timer:
    _st_dict = {}
    _ed_dict = {}
    _et_dict = {} # Elapsed time
    _setup = flag
 
    @classmethod
    def change_setup(cls, flag):
        cls._setup=flag
        # print("BREAKDOWN_FLAG setted up by change_setup()!")
 
    @classmethod
    def tick(cls, key: str):
        if not cls._setup:
            return
        torch.cuda.synchronize()
        if key not in cls._st_dict:
            cls._st_dict[key] = []
        cls._st_dict[key].append(time.time())
    
    @classmethod
    def tock(cls, key: str):
        if not cls._setup:
            return
        torch.cuda.synchronize()
        if key not in cls._ed_dict:
            cls._ed_dict[key] = []
        cls._ed_dict[key].append(time.time())
        # cls._ed_dict[key] = time.time()
 
    @classmethod
    def get_elapsed_time_ms(cls, key:str) -> float:
        if not cls._setup:
            return 0.0, 0.0
        if key not in cls._st_dict or key not in cls._ed_dict:
            print(f"{key} not measured!")
            return 0.0, 0.0
        return (cls._ed_dict[key] - cls._st_dict[key])*1000
        #print("Elapsed Time(ms) of ",key,": ",(cls._ed_dict[key] - cls._st_dict[key])*1000)
 
    @classmethod
    def get_avg_elapsed_time_ms(cls, key:str) -> tuple:
        if not cls._setup:
            return 0.0, 0.0
        if key not in cls._st_dict or key not in cls._ed_dict:
            print(f"{key} not measured!")
            return 0.0, 0.0
        
        if key not in cls._et_dict:
            cls._et_dict[key] = []

        for i in range(len(cls._ed_dict[key])):
            cls._et_dict[key].append((cls._ed_dict[key][i] - cls._st_dict[key][i]) * 1000)

        return np.mean(cls._et_dict[key]), np.std(cls._et_dict[key])
    
    @classmethod
    def print_et_times(cls, key:str):
        if not cls._setup:
            return
        if key not in cls._et_dict:
            print(f"{key} not measured!")
            return
        
        for i in range(len(cls._et_dict[key])):
            print(f"{i}: {cls._et_dict[key][i]}")


    @classmethod
    def print_keys(cls):
        if not cls._setup:
            return
        st_key_list = list(cls._st_dict.keys())
        for k in st_key_list:
            if k in cls._ed_dict:
                print(k)

    @classmethod
    def clear_time(cls):
        if not cls._setup:
            return
        cls._st_dict.clear()
        cls._ed_dict.clear()
        cls._et_dict.clear()