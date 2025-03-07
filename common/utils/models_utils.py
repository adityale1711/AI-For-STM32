from typing import Optional, Dict


def check_model_support(model_name: str, version: Optional[str], supported_models: Dict = None,
                        message: Optional[str] = None) -> None:
    if model_name not in supported_models:
        x = list(supported_models.keys())
        raise ValueError(f'\nSupported model name are {x}. received {model_name}.{message}')

    model_version = supported_models[model_name]
    if model_version:
        if not version:
            raise ValueError(f'\nMissing "version" attribute for "{model_name}" model.{message}')
        if version not in model_version:
            raise ValueError(f'\nSupported version for "{model_name}" model are {model_version}. '
                             f'Received {version}.{message}')
    else:
        if version:
            raise ValueError(f'\nThe "version" attribute is not applicable to "{model_name}" model.{message}')
