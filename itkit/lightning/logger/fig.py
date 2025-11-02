import os
from typing_extensions import override

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from matplotlib.figure import Figure



class FigureLogger(Logger):
    def __init__(self,
                 root_dir:str,
                 name:str,
                 version:str,
                 subdir:str = 'figures',
                 dpi:int = 200,
                 bbox_inches = 'tight',
                 pad_inches:int = 0,
                 *args, **kwargs):
        super().__init__()
        self._root_dir = root_dir
        self._name = name
        self._version = version
        self.subdir = subdir
        self.bbox_inches = bbox_inches
        self.pad_inches = pad_inches

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def version(self):
        return self._version

    @property
    @override
    def root_dir(self):
        return self._root_dir

    @rank_zero_only
    def log_figure(self, name:str, fig:Figure, step:int):
        assert self.version is not None
        save_dir = os.path.join(self.root_dir, self.name, self.version, self.subdir)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(os.path.join(save_dir, f"{name.replace('/', '_')}_{step}.png"),
                    dpi=200,
                    bbox_inches=self.bbox_inches,
                    pad_inches=self.pad_inches)

    def log_metrics(self, metrics, step = None) -> None:
        ...

    def log_hyperparams(self, params, *args, **kwargs) -> None:
        ...

