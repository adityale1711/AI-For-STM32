from munch import DefaultMunch
from omegaconf import DictConfig, OmegaConf


def get_config(config_data:DictConfig) -> DefaultMunch:
    config_dict = OmegaConf.to_container(config_data)
