import os
from typing import override

from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from tabulate import tabulate
from torch import Tensor
from tqdm import tqdm


class TabulateLogger(Logger):
    def __init__(self,
                 root_dir:str,
                 name:str,
                 version:str,
                 row_names:list[str],
                 column_names:list[str]):
        super().__init__()
        self._root_dir = root_dir
        self._name = name
        self._version = version
        self.row_names = row_names
        self.column_names = column_names

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
    def log_metrics(self, metrics: dict[str, Tensor|float], step: int|None = None) -> None:
        """metric naming rule: <col_name>_<row_name>"""

        if not metrics:
            return

        for k in list(metrics.keys()):
            metrics[k.split('/')[-1]] = metrics[k]
            metrics.pop(k, None)

        # try format a multi-colume tabulate to improve readability
        # majorly designed for validation results
        exist_valid_value = False
        fmt = []
        for row_name in self.row_names:
            one_row:list = [row_name]
            for col_name in self.column_names:
                v = metrics.get(f"{col_name}_{row_name}", None)
                one_row.append(v)
                if v is not None:
                    exist_valid_value = True
            fmt.append(one_row)

        if exist_valid_value:
            table = tabulate(
                fmt,
                headers=['Metric'] + self.column_names,
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
