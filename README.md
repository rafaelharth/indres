# Understanding Individual Neurons of ResNet Through Improved Compositional Formulas

## Extended Results

If you just want to look at non-cherry picked results, check out the .html files under results/. FL indicates formula length, and close/standard indicates whether the algorithm is using the new set of connectors. The browser may take a few seconds to load the files because of the large number of images.

## Setup Instructions

1. Download Broden dataset with `./script/dlbroden.sh`.
2. Run `./script/dlzoo_example.sh` to download `resnet18` trained on `places365`.
3. Ensure necessary dependencies; you should be able to clone an environment from environment.yml without complications.
4. Configure settings in `settings.py`. All settings are documented. The current settings replicate the results up to formula length 3. Set GPU to false if none is available.
5. Run `python main.py` to generate results in `results/` (the output folder name is printed when you run).

The `results/` directory contains the following files:
- `inspected_neurons.mmap`: the neuron activations across all images
- `inspected_neurons_shape.npy`: the shape of the respective array
- `static_thresholds_{quantile}.npy` the static thresholds computed for a specific quantile
- A file for label masks, e.g. `masks_MD128_X3.npy` with naming schema depending on the settings (see `settings.py`)
- An output folder for each set of settings, e.g. `FL3_imrou75_X3_close` (see `settings.py`) containing
  - The html file `{L}_{U}.html` used to view the results, where `L` and `U` are the indices of the smallest and largest neuron, respecively (ranging from 0 to 511)
  - The folder `images_{L}_{U}`
  - `history_{L}_{U}.txt` (see `jumpstart/__init__.py`)
  - The file `results_{L}_{U}.csv` which is a small table containing the essential results for these settings

## Credit

The code for this project has been originally taken from Mu and Andreas' paper [Compositional Explanations of Neurons](https://arxiv.org/abs/2006.14032), licensed under [CC-BY-SA](https://creativecommons.org/licenses/by-sa/4.0/). The code has been edited substantially, including changes to the folder structure and the names of classes, often to the point of unrecognizability. Credit is also given in the comment at the top of every class which at some point contained code from the above release.
