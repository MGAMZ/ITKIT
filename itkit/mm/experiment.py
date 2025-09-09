import os, re, pdb, glob, logging
from colorama import Fore, Style

import torch
from mmengine.logging import print_log
from mmengine.config import Config
from mmengine.runner.checkpoint import find_latest_checkpoint

from itkit.mm.mmeng_PlugIn import DynamicRunnerGenerator


class experiment:
    def __init__(self,
                 config,
                 work_dir,
                 test_work_dir,
                 cfg_options,
                 test_mode,
                 detect_anomaly,
                 test_use_last_ckpt):
        self.config = config
        self.work_dir = work_dir
        self.test_work_dir = test_work_dir
        self.cfg_options = cfg_options
        self.test_mode = test_mode
        self.detect_anomaly = detect_anomaly
        self.test_use_last_ckpt = test_use_last_ckpt
        
        with torch.autograd.set_detect_anomaly(detect_anomaly):
            self._prepare_basic_config()
            self._main_process()

    def _main_process(self):
        if self.IsTested(self.cfg):
            print_log(
                f"{Fore.BLUE}Test has been done, skipping: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

        elif self.test_mode is True:
            print_log(f"{Fore.BLUE}Test start: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)
            self._direct_to_test()
            # model_param_stat(cfg, runner)
            print_log(f"{Fore.GREEN}Test complete: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)

        elif self.IsTrained(self.cfg):
            print_log(
                f"{Fore.BLUE}Train done, please use single mode to test the model: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

        else:
            runner = DynamicRunnerGenerator(self.cfg)  # build runner
            print_log(f"{Fore.BLUE}Train start: {self.work_dir}{Style.RESET_ALL}",
                      'current', logging.INFO)
            runner.train()
            print_log(
                f"{Fore.GREEN}Train done, please use single mode to test the model: {self.work_dir}{Style.RESET_ALL}",
                'current', logging.INFO)

    def _prepare_basic_config(self):
        cfg = Config.fromfile(self.config)  # load config
        cfg.work_dir = self.work_dir  # set work dir
        if self.cfg_options is not None:
            cfg = cfg.merge_from_dict(self.cfg_options)  # cfg override
        print_log(f"Experiment work dir: {self.work_dir}", 'current', logging.INFO)
        self.cfg = cfg

    def _direct_to_test(self):
        # Check if is at multi-GPU mode
        if os.getenv('LOCAL_RANK') is not None:
            print(f"Running with torchrun. Test mode requires single GPU mode.")

        # Override configurations for testing.
        self.modify_cfg_to_skip_train()
        self.modify_cfg_to_ensure_single_node()
        self.modify_cfg_to_set_test_work_dir()

        # Initialize model.
        runner = DynamicRunnerGenerator(self.cfg)
        if self.test_use_last_ckpt:
            ckpt_path = find_latest_checkpoint(self.work_dir)
        else:
            ckpt_path = glob.glob(os.path.join(self.work_dir, 'best*.pth'))
            assert len(ckpt_path) == 1, f"Tring to find best model at {ckpt_path}, but cannot determine which is the best one."
            ckpt_path = ckpt_path[0]
        print_log(f"Loading model checkpoint: {self.work_dir}", 'current', logging.INFO)
        runner.load_checkpoint(ckpt_path)
        print_log(f"Model checkpoint loaded, doing test now: {self.work_dir}", 'current', logging.INFO)

        # execute test
        runner.test()

        # model_param_stat(cfg, runner)
        print_log(f"Test complete: {self.work_dir}", 'current', logging.INFO)

    def modify_cfg_to_skip_train(self):
        # remove train and val cfgs
        self.cfg.train_dataloader = None
        self.cfg.train_cfg = None
        self.cfg.optim_wrapper = None
        self.cfg.param_scheduler = None
        self.cfg.val_dataloader = None
        self.cfg.val_cfg = None
        self.cfg.val_evaluator = None
        self.cfg.resume = False

    def modify_cfg_to_ensure_single_node(self):
        self.cfg.launcher = 'none'
        self.cfg.model_wrapper_cfg = None
        self.cfg.strategy = None
        self.cfg.Compile = None
        self.cfg.compile = None

    def modify_cfg_to_set_test_work_dir(self):
        self.cfg.work_dir = self.test_work_dir
        self.cfg.visualizer.save_dir = self.test_work_dir

    @staticmethod
    def IsTrained(cfg) -> bool:
        if "iters" in cfg.keys():
            target_iters = cfg.iters
            work_dir_path = cfg.work_dir
            if not os.path.exists(os.path.join(work_dir_path, "last_checkpoint")):
                return False
            last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"), 'r').read()
            last_ckpt = re.findall(r"iter_(\d+)", last_ckpt)[0].strip(r'iter_')
        else:
            target_iters = cfg.epochs
            work_dir_path = cfg.work_dir
            if not os.path.exists(os.path.join(work_dir_path, "last_checkpoint")):
                return False
            last_ckpt = open(os.path.join(work_dir_path, "last_checkpoint"),
                            'r').read()
            last_ckpt = re.findall(r"epoch_(\d+)", last_ckpt)[0].strip(r'epoch_')
        if int(last_ckpt) >= target_iters:
            return True
        else:
            return False

    @staticmethod
    def IsTested(cfg: str) -> bool:
        test_file_path = os.path.join(
            cfg.work_dir,
            f"test_result_epoch{cfg.get('epoch', 0)}_iter{cfg.get('iters', 0)}.json"
        )
        if os.path.exists(test_file_path):
            return True
        else:
            return False
