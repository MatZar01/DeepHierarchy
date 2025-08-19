"""
@author: mromaszewski@iitis.pl, kfilus@iitis.pl -- OG methods
@author: mzarski@iitis.pl -- getting this mess to actually work
"""
from .args_manager import get_args, create_dir
from .print_style import Style
from .compute_SDs import compute_wn_depths, compute_net_depths
from .generate_results import get_ncsm_depths, get_ccsm_depths, save_res_yml
