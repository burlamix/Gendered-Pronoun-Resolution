
import yaml
import os

def load_logger_config_file (config_file = os.path.join(os.path.dirname(__file__), os.path.pardir, 'logging_config', 'logging.yml')):
    with open(config_file, 'r') as stream:
        config = yaml.load(stream)
    return config

