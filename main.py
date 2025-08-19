from src import get_args, create_dir
from src import Style
from src import compute_wn_depths, compute_net_depths
from src import get_ncsm_depths, get_ccsm_depths, save_res_yml


if __name__ == "__main__":
    # create working dir
    DIR_PATH = create_dir()
    # get saved config
    config = get_args()

    # FIRST do SD VALUES
    # compute depth values for WordNet
    print(f"{Style.green('WordNet')} depths:")
    compute_wn_depths(config)
    # compute values for models
    print(f"\n{Style.green('Models')} depths:")
    compute_net_depths(config)

    # DO RES GENERATION
    # compute NCSM results
    ncsm_data = get_ncsm_depths(config=config, dir_path=DIR_PATH)
    # compute CCSM results
    ccsm_data = get_ccsm_depths(config=config, dir_path=DIR_PATH)
    # save results yaml
    save_res_yml(config=config, dir_path=DIR_PATH, ncsm_data=ncsm_data, ccsm_data=ccsm_data)

