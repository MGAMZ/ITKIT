import os, re, argparse, pdb
from pathlib import Path
from bdb import BdbQuit
from os import path as osp
from colorama import Fore, Style

import torch
from mmengine.config import DictAction

from itkit.mm import MM_WORK_DIR_ROOT, MM_TEST_DIR_ROOT, MM_CONFIG_ROOT

SUPPORTED_MODELS = os.environ.get("supported_models", "").split(",")


def is_in_torch_distributed_mode():
    return "LOCAL_RANK" in os.environ


class auto_runner:
    def __init__(
        self,
        exp_names,
        model_names,
        work_dir_root,
        test_work_dir_root,
        config_root,
        cfg_options,
        test,
        auto_retry,
        detect_anomaly,
        test_use_last_ckpt,
    ):
        self.exp_names = exp_names
        self.model_names = model_names
        self.work_dir_root = work_dir_root
        self.test_work_dir_root = test_work_dir_root
        self.config_root = config_root
        self.cfg_options = cfg_options
        self.test = test
        self.auto_retry = auto_retry
        self.detect_anomaly = detect_anomaly
        self.test_use_last_ckpt = test_use_last_ckpt

    @classmethod
    def start_from_args(cls):
        parser = argparse.ArgumentParser(description="itkit OpenMM experiment runner")
        parser.add_argument("exp_name", type=str, nargs="+", help="Experiment name or version")
        parser.add_argument("--VRamAlloc", type=str, default="pytorch", help="Set memory allocator")
        parser.add_argument("--local-rank", type=int, default=0, help="Number of nodes (local rank)")
        parser.add_argument("--models", type=str, default=SUPPORTED_MODELS, help="Select models", nargs="+")
        parser.add_argument("--work-dir-root", type=str, default=MM_WORK_DIR_ROOT, help="Root directory to store experiment results")
        parser.add_argument("--test-work-dir-root", type=str, default=MM_TEST_DIR_ROOT, help="Working directory during testing")
        parser.add_argument("--config-root", type=str, default=MM_CONFIG_ROOT, help="Root directory for configuration files",)
        parser.add_argument("--cfg-options", nargs="+", action=DictAction)
        parser.add_argument("--test", default=False, action="store_true", help="Test only mode")
        parser.add_argument("--auto-retry", type=int, default=0, help="Number of auto retries for a failed experiment")
        parser.add_argument("--detect-anomaly", default=False, action="store_true", help="Enable PyTorch anomaly detection")
        parser.add_argument("--test-use-last-ckpt", default=False, action="store_true", help="Use final checkpoint instead of best during testing")
        args = parser.parse_args()
        return cls(
            exp_names=args.exp_name,
            model_names=args.models,
            work_dir_root=args.work_dir_root,
            test_work_dir_root=args.test_work_dir_root,
            config_root=args.config_root,
            cfg_options=args.cfg_options,
            test=args.test,
            auto_retry=args.auto_retry,
            detect_anomaly=args.detect_anomaly,
            test_use_last_ckpt=args.test_use_last_ckpt,
        )

    def find_full_exp_name(self, exp_name):
        if exp_name[-1] == ".":
            raise AttributeError(f"Target experiment name must not end with '.': {exp_name}")

        exp_list = os.listdir(self.config_root)
        for exp in exp_list:

            if exp == exp_name:
                print(f"Found experiment: {exp_name} <-> {exp}")
                return exp

            elif exp.startswith(exp_name):
                pattern = (
                    r"\.[a-zA-Z]"  # Regex to find the first occurrence of '.' followed by a letter
                )
                match = re.search(pattern, exp)

                if match is None:
                    raise ValueError(f"Cannot match experiment number under {self.config_root} directory: {exp}")

                if exp[: match.start()] == exp_name:
                    print(f"Found experiment by prefix: {exp_name} -> {exp}")
                    return exp

        else:
            print(f"No experiment found under {self.config_root} directory: {exp_name}")
            return None

    def experiment_queue(self):
        print("Experiment queue started, importing dependencies...")
        from itkit.mm.experiment import experiment
        
        def search_available_model_configs(exp_cfg_folder:Path):
            available_model_cfgs = [
                py_file
                for py_file in exp_cfg_folder.glob("*.py")
                if py_file.name != "mgam.py"
            ]
            if len(available_model_cfgs) == 0:
                raise FileNotFoundError(f"No available model config files found in directory: {exp_cfg_folder}")
            else:
                return available_model_cfgs

        for exp in self.exp_names:
            exp = self.find_full_exp_name(exp)
            if exp is None:
                continue
            print(f"{exp} experiment starting")

            # If no model names specified, search automatically
            for model in self.model_names or search_available_model_configs(Path(self.config_root, exp)):
                # Determine config file path and save directories
                config_path = os.path.join(self.config_root, f"{exp}/{model}.py")
                if not os.path.exists(config_path):
                    print(f"Config file does not exist: {config_path}, skipping this experiment")
                    continue
                work_dir_path = osp.join(self.work_dir_root, exp, model)
                test_work_dir_path = osp.join(self.test_work_dir_root, exp, model)

                # Set terminal title
                if os.name == "nt":
                    os.system(f"{model} - {exp} ")
                else:
                    print(f"\n--------- {model} - {exp} ---------\n")

                # Execute with auto-retry
                remain_chance = self.auto_retry + 1
                while remain_chance:
                    remain_chance -= 1

                    try:
                        experiment(
                            config_path,
                            work_dir_path,
                            test_work_dir_path,
                            self.cfg_options,
                            self.test,
                            self.detect_anomaly,
                            self.test_use_last_ckpt
                        )

                    except KeyboardInterrupt:
                        raise KeyboardInterrupt
                    except BdbQuit:
                        raise BdbQuit

                    except Exception as e:
                        if remain_chance == 0:
                            print(
                                Fore.RED
                                + f"Exception, retried {self.auto_retry} times and failed, aborting. Error reason:\n"
                                + Style.RESET_ALL,
                                e,
                            )
                            raise e
                        else:
                            print(
                                Fore.YELLOW
                                + f"Exception, remaining retry attempts: {remain_chance}, error reason:\n"
                                + Style.RESET_ALL,
                                e,
                            )

                    else:
                        print(Fore.GREEN + f"Experiment completed: {work_dir_path}" + Style.RESET_ALL)
                        if torch.distributed.is_initialized():
                            torch.distributed.destroy_process_group()
                        break


def main():
    runner = auto_runner.start_from_args()
    runner.experiment_queue()


if __name__ == "__main__":
    main()
