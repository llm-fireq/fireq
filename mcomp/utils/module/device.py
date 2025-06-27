import torch
import psutil
import pynvml
import os

def system_memory():
    
    vm = psutil.virtual_memory()
    gib = 1024**3
    return {"total_gib": vm.total / gib, "available_gib": vm.available/ gib}

def gpu_memory():
    gpus = []
    
    if torch.cuda.is_available():
        
        pynvml.nvmlInit()
        
        gib = 1024**3
        
        cuda_visible_str = os.environ.get('CUDA_VISIBLE_DEVICES')
        cuda_visible_list = []
        if cuda_visible_str is not None:
            cuda_visible_list = list(map(int, cuda_visible_str.split(',')))

        for idx in range(pynvml.nvmlDeviceGetCount()):
            
            ## Refer CUDA_VISIBLE_DEVICES
            if cuda_visible_list and idx not in cuda_visible_list:
                continue
                
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            name = pynvml.nvmlDeviceGetName(h).decode()
            
            gpus.append(
                dict(
                    id=idx,
                    name=name,
                    total_gib = info.total / gib,
                    used_gib = info.used / gib,
                    free_gib = info.free / gib,
                )
            )
            
        pynvml.nvmlShutdown()
    return gpus