import yaml
import sys
from .print_style import Style
from datetime import datetime
import os

SEP = os.sep

def get_args() -> dict:
    """returns config dict for input"""
    cfg_name = 'default'
    args = sys.argv

    if len(args) == 1:
        print(f"{Style.green('[INFO]')} no config specified, using {Style.orange(cfg_name)}")

    else:
        cfg_name = args[-1]
        print(f"{Style.green('[INFO]')} using {Style.orange(cfg_name)} config")

    try:
        config = yaml.load(open(f'.{SEP}configs{SEP}{cfg_name}.yml', 'r'), Loader=yaml.Loader)
    except FileNotFoundError:
        print(f"{Style.red('[ERROR]')} config: {Style.orange(cfg_name)} not found, aborting")
        sys.exit()

    return config


def create_dir():
    now = datetime.now()
    dir_path = f'{now.year}_{now.month}_{now.day}+{now.hour}_{now.minute}_{now.second}'
    os.mkdir(f'.{SEP}results{SEP}{dir_path}')

    return dir_path
