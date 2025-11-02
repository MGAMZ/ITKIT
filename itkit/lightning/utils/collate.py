import pdb
from collections.abc import Sequence
from torch.utils.data.dataloader import default_collate


def multi_sample_collate(data_batch: Sequence[dict]):
    """
    Compatible with `SampleAugment` Transform Class.
    This collate is to facilitate multi-sub-sample generation
    from the same sample.
    
    NOTE
    The reason to do SampleWiseInTimeAugment is the time comsumption
    for IO of an entire sample is too expensive, so it's better
    to augment the sample in time, thus accquiring multiple trainable sub-samples.
    """
    flattened = []
    for item in data_batch:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
            
    return default_collate(flattened)



