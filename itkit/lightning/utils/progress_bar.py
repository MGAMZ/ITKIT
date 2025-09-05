from pytorch_lightning.callbacks.progress import ProgressBar


def mgam_bar(lightning_progress_bar: type[ProgressBar]):
    class CustomizedProgressBar(lightning_progress_bar):
        def get_metrics(self, trainer, pl_module):
            # don't show the version number
            items = super().get_metrics(trainer, pl_module)
            items.pop("v_num", None)
            return items
    return CustomizedProgressBar
