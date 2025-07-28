import os, pdb
from typing_extensions import override
from tqdm import tqdm

from torch import Tensor
from tabulate import tabulate
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import Logger



class TabulateLogger(Logger):
    def __init__(self,
                 root_dir:str,
                 name:str,
                 version:str,):
        super().__init__()
        self._root_dir = root_dir
        self._name = name
        self._version = version

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

    @override
    @rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: dict[str, Tensor|float], step: int|None = None
    ) -> None:
        if not metrics:
            return
        
        for k in list(metrics.keys()):
            metrics[k.split('/')[-1]] = metrics[k]
            metrics.pop(k, None)

        # 保持顺序的去重
        criterion_names = list(dict.fromkeys([k.split('_')[0] for k in metrics.keys()]))
        class_names = list(dict.fromkeys(['_'.join(k.split('_')[1:]) for k in metrics.keys()]))
        if '' in criterion_names:
            criterion_names.remove('')
        if '' in class_names:
            class_names.remove('')

        # try format a multi-colume tabulate to improve readability
        # majorly designed for validation results
        fmt = []
        for class_name in class_names:
            fmt_row:list = [class_name]
            for criterion in criterion_names:
                row_name = f'{criterion}_{class_name}'
                if row_name in metrics:
                    fmt_row.append(metrics[row_name])
            fmt.append(fmt_row)
            
        table = tabulate(
            fmt,
            headers=['Metric'] + list(criterion_names),
            tablefmt='grid',
            floatfmt='.3f',
        )

        tqdm.write(table)
        with open(os.path.join(self.root_dir, self.name, self.version, 'tabulates.txt'), "a") as f:
            f.write(table + "\n")
            if step is not None:
                f.write(f"Step: {step}\n")
            f.write("\n")

    @override
    @rank_zero_only
    def log_hyperparams(self, params=None) -> None:
        ...
