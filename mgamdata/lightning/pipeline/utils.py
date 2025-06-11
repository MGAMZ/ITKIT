from .base import BaseTransform


class TypeConvert(BaseTransform):
    def __init__(self, key:str|list[str], dtype):
        self.key = key if isinstance(key, list) else [key]
        self.dtype = dtype
    
    def __call__(self, sample:dict) -> dict:
        for k in self.key:
            sample[k] = sample[k].astype(self.dtype)
        return sample
