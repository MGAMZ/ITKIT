from pytorch_lightning.loggers import TensorBoardLogger as _TensorBoardLogger


class TensorBoardLogger(_TensorBoardLogger):
    def log_figure(self, name:str, fig, step:int):
        self.experiment.add_figure(name, fig, step)
