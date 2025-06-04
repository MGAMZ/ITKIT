import pdb
import timm
import torch


class timm_base_model(torch.nn.Module):
    def __init__(self, timm_create_model_args:dict):
        super(timm_base_model, self).__init__()
        self.model = timm.create_model(**timm_create_model_args)
    
    def forward(self, x:torch.Tensor):
        pdb.set_trace()
        return self.model(x)
