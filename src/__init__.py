"""
@author: mromaszewski@iitis.pl, kfilus@iitis.pl -- OG methods
@author: mzarski@iitis.pl -- getting this to work
"""
from .args_manager import get_args, create_dir
from .print_style import Style
from .compute_SDs import compute_wn_depths, compute_net_depths
from .generate_results import get_ncsm_depths, get_ccsm_depths, save_res_yml
from .check_compliance import get_dir_compliance, get_ext_compliance, plot_dir_compliance, plot_ext_compliance
from .augments import Augmenter
