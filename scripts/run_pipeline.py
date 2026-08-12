"""Run one pipeline rollout from the command line.

Examples:
    python scripts/run_pipeline.py env=vega_motion3d mode=fake
    python scripts/run_pipeline.py env=vega_motion3d mode=sim
    python scripts/run_pipeline.py env=vega_motion3d mode=real  # on the robot
"""

import hydra
from omegaconf import DictConfig

from prpl_dexmate.pipeline import run_pipeline


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def _main(cfg: DictConfig) -> None:
    summary = run_pipeline(cfg)
    print(summary)


if __name__ == "__main__":
    _main()  # pylint: disable=no-value-for-parameter
