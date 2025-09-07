import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config1", type=str, default="example_ppo_megatron_critic_mini", help="Name of the PPO configuration."
)
parser.add_argument(
    "--config2", type=str, default="example_ppo", help="Name of the PPO configuration."
)
args = parser.parse_args()


def make_ppo_config():

    config_path = "."
    config_name_1 = args.config1

    initialize(config_path=config_path)
    cfg = compose(config_name=config_name_1)
    ppo_config1 = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))
    
    config_name_2 = args.config2

    # initialize(config_path=config_path)
    cfg = compose(config_name=config_name_2)
    ppo_config2 = from_dict(data_class=RLVRConfig, data=OmegaConf.to_container(cfg, resolve=True))

    return ppo_config1, ppo_config2


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():

    ppo_config1, ppo_config2 = make_ppo_config()

    init()

    from tests.pipeline.rlvr_pipeline_meg_critic import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config1=ppo_config1, pipeline_config2=ppo_config2)

    pipeline.run()


if __name__ == "__main__":
    test_ppo_pipeline()
