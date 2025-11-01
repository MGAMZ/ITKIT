import pdb
import timm
import torch


class timm_base_model(torch.nn.Module):
    def __init__(
        self, timm_create_model_args: dict, output_linear_args: dict | None = None
    ):
        super(timm_base_model, self).__init__()
        self.model = timm.create_model(**timm_create_model_args)
        if output_linear_args is not None:
            self.out_proj = torch.nn.Linear(**output_linear_args)

    def forward(self, x: torch.Tensor):
        return self.model(x)
