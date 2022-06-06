import torch
import torch.nn as nn

from utils.functions import weights_init


def get_model_pack(cfg):
    model_pack = {}
    
    from models.nerface.base_function import get_embedding_function
    #XXX Nerface == Nerface with geometry change -> 일단 기존 모델이 안되는 걸 보여야해서 실험 제일 먼저 해야함
    #XXX Nerface_A == Nerface_A with appearance change -> 레이턴트 위치를 그냥 color뽑을 때만 사용되게
    #XXX Nerface_W_transient == Nerface + NeRF-W -> 위에꺼 다하고 실험해보는게 좋을듯~

    # Positional Encoding - position / direction
    encode_position = get_embedding_function(num_encoding_functions=10, include_input=True, log_sampling=True)
    encode_direction = get_embedding_function(num_encoding_functions=4, include_input=False, log_sampling=True)
    
    if cfg.method == 'nerface':
        from models.nerface.model import Nerface   
        # Coarse & Fine Model
        model_coarse = Nerface(num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False)
        model_fine = Nerface(num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False)

    elif cfg.method == 'nerface_no_expr':
        from models.nerface.model import Nerface_no_expr

        model_coarse = Nerface_no_expr(num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False)
        model_fine = Nerface_no_expr(num_encoding_fn_xyz=10, num_encoding_fn_dir=4, include_input_dir=False)
    
        
    latent_codes = nn.Parameter(torch.zeros(cfg.dataset['data_size'], 32))

    

    model_pack['latent_codes'] = latent_codes

    model_pack['encode_position'] = encode_position
    model_pack['encode_direction'] = encode_direction
    
    model_pack['model_coarse'] = model_coarse
    model_pack['model_fine'] = model_fine

    return model_pack