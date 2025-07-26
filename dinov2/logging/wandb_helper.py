from typing import Dict, Any
import logging

import wandb
import torch.distributed as dist
from tqdm import tqdm

logger = logging.getLogger("dinov2")


def is_main_process() -> bool:
    if dist is not None and dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True  # Not a distributed job, assume main process


class WandbMetric:
    def __init__(self, config, total_iterations: int, print_every_n_steps: int = 10):
        self.is_main = is_main_process()
        self.project = config['train']['project']
        self.name = config['train']['name']
        self._wandb_run = None
        self._print_every_n_steps = print_every_n_steps
        self._total_iterations = total_iterations
        # print the progress bar only on the main process every 10 seconds
        self._bar = tqdm(
            total=total_iterations, desc='Training', smoothing=0.1,
            disable=not self.is_main, mininterval=10.,
        )

        if self.is_main:
            self._wandb_run = wandb.init(project=self.project, name=self.name, config=config)
            logger.warning("Initialized W&B logging on main process.")
        else:
            logger.warning("Skipping W&B init on non-main process.")

    def log(self, metrics: Dict[str, Any], step):
        if self.is_main:
            wandb.log(metrics, step=step)
            self._bar.update()

    def finish(self):
        if self.is_main:
            wandb.finish()
