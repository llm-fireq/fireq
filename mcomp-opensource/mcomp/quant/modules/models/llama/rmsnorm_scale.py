import torch
from dataclasses import dataclass
from mcomp.utils import find_module_by_exactmatching

@dataclass
class LlamaScales:

    attn_ln_factor:float
    v_proj_scale:float
    o_proj_scale:float
    attn_scale:float

    mlp_ln_factor:float
    up_proj_scale:float
    down_proj_scale:float
    mlp_scale:float

    def give_max_layer_scales(self):
        return max(self.attn_scale, self.mlp_scale)
    

def is_scale_available(tg_lb, proj_name, scale):
    if tg_lb[proj_name]:
        
        for _s, _e in tg_lb[proj_name]:
            if int(scale) == int(_s):
                return True
    else:
        if int(scale) == 1:
            return True
    return False

def get_layer_llamascale(tg_lb):
    min_best = {}
    for k,v in tg_lb.items():
        if v:
            min_best[k] = v[0][0]
        else:
            min_best[k] = 1.0

    ## Attn
    attn_pre_ln_factor = max(min_best['q_proj'],min_best['k_proj'])
    while attn_pre_ln_factor > 1:
        if is_scale_available(tg_lb, 'q_proj', attn_pre_ln_factor) and is_scale_available(tg_lb, 'k_proj', attn_pre_ln_factor):
            break
    
        attn_pre_ln_factor //= 2
    
    # v_proj
    v_proj_scale = 1.0
    if attn_pre_ln_factor > min_best['v_proj'] and is_scale_available(tg_lb, 'v_proj', attn_pre_ln_factor):
            v_proj_scale = attn_pre_ln_factor
    else:
        v_proj_scale =  min_best['v_proj'] / attn_pre_ln_factor
    
    o_proj_scale = min_best['o_proj']
    attn_scale = v_proj_scale*o_proj_scale

    ## MLP
    mlp_pre_ln_factor = max(min_best['up_proj'],min_best['gate_proj'])
    while mlp_pre_ln_factor > 1:
        if is_scale_available(tg_lb, 'up_proj', mlp_pre_ln_factor) and is_scale_available(tg_lb, 'gate_proj', mlp_pre_ln_factor):
            break
    
        mlp_pre_ln_factor //= 2
    
    # v_proj
    up_proj_scale = 1.0
    if mlp_pre_ln_factor > min_best['up_proj'] and is_scale_available(tg_lb, 'up_proj', mlp_pre_ln_factor):
            up_proj_scale = mlp_pre_ln_factor
    else:
        up_proj_scale = min_best['up_proj'] / mlp_pre_ln_factor
    
    down_proj_scale = min_best['down_proj']
    mlp_scale = up_proj_scale*down_proj_scale

    return LlamaScales(attn_pre_ln_factor, v_proj_scale, o_proj_scale, attn_scale, mlp_pre_ln_factor, up_proj_scale, down_proj_scale, mlp_scale)

def get_best_memory(linear, group_size = 128):
    weight = linear.weight.data
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    fp8_min = 2**-9
    bit_width = 4
    weight_min = fp8_min * (2**(bit_width-1) - 1)
    
    @torch.no_grad()
    def slack_sum(w, scale, group_size):
        gwa = (w*scale).view(-1, group_size).abs().amax(dim=1)
        gwa = gwa.to('cuda')
        # gw = w.view(-1, group_size)
        # gwa = gw.abs()
        # gwa = gwa.amax(dim=1)
        return ((weight_min - gwa[gwa < weight_min]).sum()/weight_min).to('cpu'), ((gwa[gwa > fp8_max] - fp8_max).sum()/(fp8_max)).to('cpu')
    
    best_memory = []    
    initial_list = []
    for i in range(11): ## 
        _lower, _upper = slack_sum(weight, float(2**i), group_size)
        if _upper > 0: break
        initial_list.append((torch.tensor(float(2**i)).item(), _lower.item()))
    best_memory.extend(initial_list)
    if best_memory:
        best_memory.sort(key = lambda x: (x[1], x[0]))
    return best_memory

def get_layer_bests(layers):
    layer_bests = [{} for _ in range(32)]
    _proj_keys = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj']
    for idx, layer in enumerate(layers):
        for key in _proj_keys:
            proj = find_module_by_exactmatching(layer, key)
            layer_bests[idx][key] = get_best_memory(proj, group_size=128)
            
    return layer_bests
        

def get_layer_scales(layer_bests):
    layer_scales = []
    for tg_lb in layer_bests:
        layer_scales.append(get_layer_llamascale(tg_lb))
    return layer_scales

def get_total_scale(layer_scales):
    total_max = 1.0
    for scales in layer_scales:
        candii = scales.give_max_layer_scales()
        if candii > total_max:
            total_max=  candii

    return total_max

# total_max_scale = get_total_scale(layer_scales)

def validate_and_update_layer_scales(layer_scales, layer_bests):
    total_max_scale = get_total_scale(layer_scales)
    for idx in range(len(layer_scales)):
        tg_ls = layer_scales[idx]
        tg_lb = layer_bests[idx]
        additional = total_max_scale/tg_ls.attn_scale
        while additional > 1:
            if tg_ls.v_proj_scale >= tg_ls.o_proj_scale:
                if is_scale_available(tg_lb, 'v_proj', tg_ls.v_proj_scale*2):
                    tg_ls.v_proj_scale *= 2
                elif is_scale_available(tg_lb, 'o_proj', tg_ls.o_proj_scale*2):
                    tg_ls.o_proj_scale *= 2
            else:
                if is_scale_available(tg_lb, 'o_proj', tg_ls.o_proj_scale*2):
                    tg_ls.o_proj_scale *= 2
                elif is_scale_available(tg_lb, 'v_proj', tg_ls.v_proj_scale*2):
                    tg_ls.v_proj_scale *= 2
                else:
                    break
            tg_ls.attn_scale *= 2
            additional //= 2
        
        while additional > 1:
            if is_scale_available(tg_lb, 'q_proj', tg_ls.attn_ln_factor*2) and is_scale_available(tg_lb, 'k_proj', tg_ls.attn_ln_factor*2):
                tg_ls.attn_ln_factor *= 2
                tg_ls.v_proj_scale *= 2
                tg_ls.attn_scale *= 2
                additional //= 2
            else:
                break
        
        assert additional < 2, "RMSNorm-backed Weight Amplification can not be applied"
        
        additional = total_max_scale/tg_ls.mlp_scale
        while additional > 1:
            if tg_ls.down_proj_scale >= tg_ls.up_proj_scale:
                if is_scale_available(tg_lb, 'v_proj', tg_ls.down_proj_scale*2):
                    tg_ls.down_proj_scale *= 2
                elif is_scale_available(tg_lb, 'o_proj', tg_ls.up_proj_scale*2):
                    tg_ls.up_proj_scale *= 2
            else:
                if is_scale_available(tg_lb, 'o_proj', tg_ls.up_proj_scale*2):
                    tg_ls.up_proj_scale *= 2
                elif is_scale_available(tg_lb, 'v_proj', tg_ls.down_proj_scale*2):
                    tg_ls.down_proj_scale *= 2
                else:
                    break
            tg_ls.mlp_scale *= 2
            additional //= 2
        
        while additional > 1:
            if is_scale_available(tg_lb, 'gate_proj', tg_ls.mlp_ln_factor*2):
                tg_ls.mlp_ln_factor *= 2
                tg_ls.up_proj_scale *= 2
                tg_ls.mlp_scale *= 2
                additional //= 2
            else:
                break
        
        assert additional < 2, "RMSNorm-backed Weight Amplification can not be applied"
        
### 모두 Quantized된 관점에서 다시 짜자.
















