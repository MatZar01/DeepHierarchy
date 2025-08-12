from src import get_args
from src import Style
from src import compute_wn_depths, compute_net_depths

if __name__ == "__main__":
    # get saved config
    config = get_args()

    # FIRST do SD VALUES
    # compute depth values for WordNet
    print(f"{Style.green('WordNet')} depths:")
    compute_wn_depths(config)
    # compute values for models
    print(f"{Style.green('Models')} depths:")
    compute_net_depths(config)

