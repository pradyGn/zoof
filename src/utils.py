from config import zoofv1Config


def config_dataclass(config_dict):
    return zoofv1Config(**config_dict)
