
import torch
import random
import logging
import numpy as np


class InfiniteDataLoader(torch.utils.data.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_iterator = super().__iter__()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            batch = next(self.dataset_iterator)
        except StopIteration:
            self.dataset_iterator = super().__iter__()
            batch = next(self.dataset_iterator)
        return batch


def make_deterministic(seed: int = 0):
    """Make results deterministic. If seed == -1, do not make deterministic.
        Running your script in a deterministic way might slow it down.
        Note that for some packages (eg: sklearn's PCA) this function is not enough.
    """
    seed = int(seed)
    if seed == -1:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_cudnn(benchmark: bool = False):
    if benchmark:
        # If speed is requested:
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False
        logging.info("cuDNN benchmark ENABLED: Training will be FASTER but not bit-by-bit reproducible.")
    else:
        # If reproducibility is requested (already set to False by make_deterministic):
        # This ensures exact results if the same seed is used
        logging.info("cuDNN benchmark DISABLED: Training will be bit-by-bit DETERMINISTIC (Slower).")
