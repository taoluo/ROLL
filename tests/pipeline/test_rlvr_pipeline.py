import argparse

from dacite import from_dict
from hydra.experimental import compose, initialize
from omegaconf import OmegaConf

from roll.distributed.scheduler.initialize import init
from roll.pipeline.rlvr.rlvr_config import RLVRConfig

parser = argparse.ArgumentParser(description="PPO Configuration")

parser.add_argument(
    "--config_name", type=str, default="rlvr_megatron_config", help="Name of the PPO configuration."
)
args = parser.parse_args()


def make_ppo_config():
    config_path = "."
    config_name = args.config_name

    print(f"DEBUG: Loading config_name = '{config_name}'")
    initialize(config_path=config_path)
    cfg = compose(config_name=config_name)

    # Debug: Check what's in the raw config
    print(f"DEBUG: Raw cfg.adv_estimator = '{cfg.get('adv_estimator', 'NOT_FOUND')}'")
    print(f"DEBUG: Raw cfg keys: {list(cfg.keys())}")

    # Convert to container
    cfg_container = OmegaConf.to_container(cfg, resolve=True)
    print(f"DEBUG: Container adv_estimator = '{cfg_container.get('adv_estimator', 'NOT_FOUND')}'")

    ppo_config = from_dict(data_class=RLVRConfig, data=cfg_container)
    print(f"DEBUG: Final ppo_config.adv_estimator = '{ppo_config.adv_estimator}'")

    # TEMPORARY FIX: Explicitly set adv_estimator from config if dacite is not working
    if 'adv_estimator' in cfg_container:
        print(f"DEBUG: Manually overriding adv_estimator to '{cfg_container['adv_estimator']}'")
        ppo_config.adv_estimator = cfg_container['adv_estimator']

    return ppo_config


def test_make_ppo_config():
    ppo_config = make_ppo_config()
    print(ppo_config)


def test_ppo_pipeline():
    ppo_config = make_ppo_config()
    print(f"DEBUG: After loading config, adv_estimator = '{ppo_config.adv_estimator}'")

    init()

    from roll.pipeline.rlvr.rlvr_pipeline import RLVRPipeline
    pipeline = RLVRPipeline(pipeline_config=ppo_config)

    pipeline.run()

    print("RLVR Pipeline test completed successfully.")


if __name__ == "__main__":
    test_ppo_pipeline()
