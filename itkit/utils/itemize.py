import numpy as np
from torch import Tensor


def to_item(obj):
    if isinstance(obj, (Tensor, np.ndarray)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_item(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        converted = [to_item(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(converted)
        if isinstance(obj, set):
            return set(converted)
        return converted
    return obj
