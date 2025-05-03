# from .sokoban import SokobanEnv,SokobanEnvConfig
# from .frozenlake import FrozenLakeEnv,FrozenLakeEnvConfig, FrozenLakeService
# from .navigation import NavigationEnv, NavigationEnvConfig, NavigationServiceConfig, NavigationService
# from .svg import SVGEnv, SvgEnvConfig, SVGService
# from .Embench_new import AlfredEnv, AlfredService
import importlib
from .base.base_service_config import BaseServiceConfig

REGISTERED_ENV = {
    "alfred": {
        "env_cls": ".Embench_new:AlfredEnv",
        "service_cls": ".Embench_new:AlfredService",
        "config_cls": ".Embench_new:AlfredEnvConfig",
    },
    "eb-manipulation": {
        "env_cls": ".Embench_new:EBManipulationEnv",
        "service_cls": ".Embench_new:EBManipulationService",
        "config_cls": ".Embench_new:EBManipulationEnvConfig",
    },
    "frozenlake": {
        "env_cls": ".frozenlake:FrozenLakeEnv",
        "service_cls": ".frozenlake:FrozenLakeService",
        "config_cls": ".frozenlake:FrozenLakeEnvConfig",
    },
}

def load_class(path:str, package=__package__):
    """
    Given a string of the form "module.submod:ClassName",
    import module.submod and return the attribute ClassName.
    """
    mod_path, cls_name = path.split(":")
    module = importlib.import_module(mod_path, package=package)
    return getattr(module, cls_name)

def get_env(name):
    if name not in REGISTERED_ENV:
        raise ValueError(f"Environment '{name}' is not registered.")
    if "env_cls" not in REGISTERED_ENV[name]:
        raise ValueError(f"Environment '{name}' does not have an environment class registered.")
    cfg = REGISTERED_ENV[name]
    return load_class(cfg["env_cls"])

def get_service(name):
    if name not in REGISTERED_ENV:
        raise ValueError(f"Environment '{name}' is not registered.")
    if "service_cls" not in REGISTERED_ENV[name]:
        raise ValueError(f"Environment '{name}' does not have a service class registered.")
    cfg = REGISTERED_ENV[name]
    return load_class(cfg["service_cls"])

def get_config(name):
    if name not in REGISTERED_ENV:
        raise ValueError(f"Environment '{name}' is not registered.")
    if "config_cls" not in REGISTERED_ENV[name]:
        raise ValueError(f"Environment '{name}' does not have a config class registered.")
    cfg = REGISTERED_ENV[name]
    return load_class(cfg["config_cls"])


def get_service_config(name):
    if name not in REGISTERED_ENV:
        raise ValueError(f"Environment '{name}' is not registered.")
    if "service_config_cls" not in REGISTERED_ENV[name]:
        return BaseServiceConfig
    cfg = REGISTERED_ENV[name]
    return load_class(cfg["service_config_cls"])

def get_all_envs():
    """
    Returns a list of all registered environments.
    """
    return list(REGISTERED_ENV.keys())