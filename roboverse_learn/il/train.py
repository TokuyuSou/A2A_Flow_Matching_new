import os
import pathlib
import sys

import hydra
import torch.distributed as dist
from omegaconf import OmegaConf

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
sys.path.insert(0, project_root)
from roboverse_learn.il.runners.base_runner import BaseRunner

abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("configs").absolute())
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path=abs_config_path, version_base="1.3")
def main(cfg):
    OmegaConf.resolve(cfg)

    # Initialize distributed training when launched via torchrun
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")

    cls = hydra.utils.get_class(cfg._target_)
    runner: BaseRunner = cls(cfg)

    try:
        runner.run()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
