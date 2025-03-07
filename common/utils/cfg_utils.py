from munch import DefaultMunch
from typing import Dict, List


def get_random_seed(global_seed):
    if global_seed is not None:
        seed = global_seed
        if seed == 'None':
            seed = None
        else:
            seed = int(seed)
    else:
        seed = 0

    return seed


def check_attributes(cfg: Dict, expected: List[str] = None, optional: List[str] = [], section: str = None) -> None:
    if section is not None:
        message = f'\nPlease check the "{section}" section of your configuration file.'
    else:
        message = f'\nPlease check your configuration file.'

    if cfg is not None:
        if type(cfg) is not DefaultMunch:
            raise ValueError(f'Expecting an attribute. Received {cfg}{message}')

        # Check that each attribute name is legal
        for attr in cfg.keys():
            if (not attr in expected) and (not attr in optional):
                raise ValueError(f'\nUnknown or unsupported attribute. Received {attr}{message}')

        # Get the list of used attributes
        used = list(cfg.keys())
    else:
        used = []

    # Check that all the mandatory attributes are present
    for attr in expected:
        if attr not in used:
            raise ValueError(f'\nMissing "{attr}" attribute{message}')
        if cfg[attr] is None:
            raise ValueError(f'\nMissing a value for attributes "{attr}"{message}')
