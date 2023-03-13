import torch
import torch.nn as nn

from transformers import MT5Config, MT5ForConditionalGeneration
from utils import expand_to_batchsize_for_layer
from functools import partial

class PrefixTuning(nn.Module):
    """
    """
    def __init__(self, model, preseqlen=5, hidden_dim=512):
        super().__init__()

        raw_embedding = model.get_input_embeddings()
        self.config = model.config
        self.mapping_hook = None
        self.embedding_size = raw_embedding.weight.shape[-1]
        self.num_token = preseqlen

        self.using_encoder_past_key_values = True
        self.using_decoder_past_key_values = False

        if isinstance(self.config, MT5Config):
            self.n_layer = self.config.num_layers
            self.n_embd = self.config.d_model
            self.n_head = self.config.num_heads
            self.n_decoder_layer = self.config.num_decoder_layers
            self.match_n_decoder_layer = self.n_decoder_layer
            self.match_n_layer = self.n_layer
       
        self.mid_dim = hidden_dim
        self.match_n_head = self.n_head
        # self.match_n_embd = self.n_embd // self.n_head
        # Hardcode this as the documentation of mT5 is inconsistent and uses 64 instead of the 'original' way on line 33
        self.match_n_embd = 64
        self.prefix_dropout = 0.0
        self.dropout = nn.Dropout(self.prefix_dropout)

        self.past_key_values = None
        
        self.generate_parameters() # in prefix tuning the template text has no interact with the parameters.

        self.plm_modified = False # flag to indicate whether the function of plm are replaced for prefix tuning.
        self.model = self.modify_plm(model)
        
        for name, param in self.model.named_parameters():                
            param.requires_grad = False
        
    def forward(self, batch_size=4):
        pvs = []
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            input_tokens = self.input_tokens.unsqueeze(0).expand(batch_size, -1)
            temp_control = self.wte(input_tokens)
            past_key_values = self.control_trans(temp_control) #bsz, seqlen, layer*emb
            _, seqlen, _ = past_key_values.shape
            past_key_values = past_key_values.view(batch_size, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            pvs.append(past_key_values)
        else:
            pvs.append(None)

        return pvs

    def generate_parameters(self) -> None:
        r"""
        Generate parameters needed for new tokens' embedding in P-tuning
        """
        
        self.input_tokens = nn.Parameter(torch.arange(self.num_token).long(), requires_grad=False) # to allow automatic devicing
        if self.config.is_encoder_decoder and self.using_encoder_past_key_values:
            self.wte = nn.Embedding(self.num_token, self.n_embd)
            self.control_trans = nn.Sequential(
                nn.Linear(self.n_embd, self.mid_dim),
                nn.Tanh(),
                # nn.Linear(self.mid_dim, self.mid_dim),
                # nn.Tanh(),
                nn.Linear(self.mid_dim, 6144))

    def modify_plm(self, model):
        if self.plm_modified:
            return None
        if isinstance(model, MT5ForConditionalGeneration):
            backup_encoder_forward_functions = []
            for i, layer_module in enumerate(model.encoder.block):
                backup_encoder_forward_functions.append(layer_module.layer[0].forward)
                def modified_encoder_forward(*args, **kwargs):
                    layer_id = kwargs.pop('layer_id')
                    batch_size = args[0].shape[0]
                    device = args[0].device
                    if kwargs['past_key_value'] is None:
                        kwargs['past_key_value'] = expand_to_batchsize_for_layer(self.past_key_values[0], batch_size, layer_id).to(device)
                    if kwargs['attention_mask'] is not None:
                        am = kwargs['attention_mask']  # Should check the format of the attention_mask when moving to a new plm.
                        kwargs['attention_mask'] = torch.cat([-torch.zeros((*am.shape[:-1], self.num_token), dtype=am.dtype, device=am.device), am], dim=-1)
                    return backup_encoder_forward_functions[layer_id](*args, **kwargs)
                layer_module.layer[0].forward = partial(modified_encoder_forward, layer_id=i)
        else:
            raise NotImplementedError
        self.plm_modified = True
        return model

        