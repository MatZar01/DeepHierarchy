# Deep Hierarchy

--Add description here--

## Usage

Here's how to use this tool

1. Prepare your models outputs and put them into `Models_CSM` directory in a structure: `Models_CSM/<<model_name>>/`. Required items are: `binary_matrices.pkl`, `binary_matrices_after_epoch_confusion_CCSM.pkl`, `binary_matrices_after_epoch_weights_NCSM.pkl` and `results.yml`.
2. Make sure your `WordNet_data` contains items: `Native.pkl`, `WUP.pkl` and `wordnet_classes.yml`. They are provided with this repository, but check just in case.
3. Prepare your config file and put it into `configs` directory.

Or use `default.yml` and specify within:

- `WN_PATH` -- path to WordNet data
- `MODEL_PATH` -- path to your model data (in both cases you can specify another directory that is not default to this project)
- `CHECK_EPOCH` -- specify the number of epoch you want to check the communities for (by default it is epoch 200, but you can see the communities forming across any epoch you pick)
- `SEED` -- specify random seed for graph generation (or leave it at 42)

4. Run program with `python3 main.py` if you want to use `default` config or `python3 main.py config_name` if you want to use `config_name` for the input parameters.
5. You'll get the results in `results` directory in a form:

- `MAIN_DIR` with the name corresponding to the time of running the program
- `MAIN_DIR/results_epoch_num.csv` with overall results for every model specified for given epoch.
- `MAIN_DIR/DF` with `results.csv` for every model, epoch specified, and both NCSM and CCSM similarity measurements.
- `MAIN_DIR/PLOTS` with `.png` plot of every community, for every model, epoch specified, and both NCSM and CCSM similarity measurements.
- `MAIN_DIR/DIR_COMP` and `MAIN_DIR/EXT_COMP` for compliance graphs for both direct and extended neighborhoods.

Also be sure to watch the terminal output, as it also gives out results in a straightforward way.
