import torch
import gc
from mcomp.utils.quant import (
    pseudo_group_quantize,
)

class WeightClippingMixin:

    def __init__(self): #, weight, n_bit, group_size, scales_dt):
        # self.weight = weight
        # self.n_bit = n_bit
        # self.group_size = group_size
        # self.scales_dt = scales_dt
        pass
        
    @classmethod
    def get_best_clip(cls,
                      w, 
                      x, 
                      nbits=4, 
                      group_size=128, 
                      scales_dt=torch.float8_e4m3fn,
                      device='cuda', 
                      w_batch_size=64, 
                      x_batch_size=1024, 
                      num_steps=10, 
                      max_sample_num=None):

        D = x.shape[-1]
        x = x.view(-1, D )
        BS = x.shape[0]
    
        if max_sample_num is not None:
            idx = torch.randperm(BS, device=x.device)
            x = x.index_select(dim=0, index=idx)
            min_sample_num = min(BS, max_sample_num)
            x = x[:min_sample_num]
            BS = x.shape[0]
    
        x = x.view(1, BS, -1, group_size)
        M, K = w.shape
    
        w = w.view(M, 1, -1, group_size)
        
        best_max_list = []
        for idx in range((M + w_batch_size -1) // w_batch_size):
            w_start_idx = idx*w_batch_size
            w_end_idx = min(w_start_idx + w_batch_size, M)
        
            _w = w[w_start_idx: w_end_idx]
        
            max_val = _w.abs().amax(dim=-1, keepdim=True)
            
            best_max = max_val.clone()
            min_errs = torch.ones_like(best_max) * 1e9
        
            for i in range(num_steps, 0, -1):
                alpha = (num_steps*3+i)/(num_steps*4)
        
                cur_max_val = alpha*max_val
                __w = torch.clamp(_w, min=-cur_max_val, max=cur_max_val)
                deq_w, _, _ = pseudo_group_quantize(__w, nbits, group_size=group_size, scales_dt=scales_dt)
                
                tot_err = None
                for x_idx in range((BS + x_batch_size -1)//x_batch_size):
                    x_start_idx = x_idx*x_batch_size
                    x_end_idx = min(x_start_idx + x_batch_size, BS)
                    _x = x[:, x_start_idx:x_end_idx]
                    _x = _x.to(device)
            
                    org_out = (_x*_w).sum(dim=-1)
                    cur_out = (_x*deq_w).sum(dim=-1)
                    cur_err = (org_out - cur_out).to(torch.float).pow(2).sum(dim=1).view(min_errs.shape)
                    if tot_err is None:
                        tot_err = cur_err
                    else:
                        tot_err = torch.cat([tot_err, cur_err], dim=1).sum(dim=1,keepdim=True)
            
                tot_err = tot_err / BS
                    
                del cur_out
            
                cur_best_idx = tot_err < min_errs
                min_errs[cur_best_idx] = tot_err[cur_best_idx].to(min_errs.dtype)
                best_max[cur_best_idx] = cur_max_val[cur_best_idx]
            
            best_max_list.append(best_max)
        
        best_max = torch.cat(best_max_list, dim=0)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return best_max.squeeze(1).squeeze(-1)
    
    @classmethod
    def apply_best_clip(cls, weight, best_clip): # symmetric quant

        assert weight.shape[-1] % best_clip.shape[-1] == 0
        group_size = weight.shape[-1] // best_clip.shape[-1]
        best_clip_ext = best_clip.repeat_interleave(group_size, dim=-1)
        
        weight = torch.clamp(weight, min=-best_clip_ext, max=best_clip_ext)
        return weight

                