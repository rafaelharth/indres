"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Contains various code that doesn't neatly fit into any other package.
"""

import os
import random
import re

import numpy as np
import tqdm
import S
import settings
import util.vecquantile as Vecquantile

def flatten(t):
    return [item for sublist in t for item in sublist]

def create_directories(directory_names):
    for directory_name in directory_names:
        if not os.path.exists(directory_name):
            os.mkdir(directory_name)

def sample_with_duplicates_if_necessary(population, sample):
    while len(population) < sample:
        i = random.sample(population, 1)[0]
        population.append(i)

    return random.sample(population, sample)

def purify_filename(filename):
    """Sometimes a blob name will have delimiters in it which are not
    filename safe.  Fix these by turning them into hyphens."""
    if filename is None:
        return None
    else:
        return re.sub("[-/#?*!\s]+", "-", filename).strip("-")

def compute_static_thresholds(map_n_im_2_activations):
    """
    Determine thresholds for neuron activations for each neuron.
    """
    path = f"results/static_thresholds_{settings.QUANTILE}.npy"

    if os.path.exists(path):
        return np.load(path)

    quantile_vector = Vecquantile.QuantileVector(depth=S.n_neurons, seed=1)
    batch_size = 64

    for im_i in tqdm.trange(0, S.n_images, batch_size, desc="Computing static thresholds"):
        batch = map_n_im_2_activations[:, im_i : im_i + batch_size] # neuron, image (100), y, x
        batch = np.transpose(batch, axes=(1, 2, 3, 0)).reshape(-1, S.n_neurons) # image_flattened, neuron
        quantile_vector.add(batch)

    quantiles = quantile_vector.readout(1000)
    thresholds = quantiles[:, int(1000 * (1 - settings.QUANTILE) - 1)]

    np.save(path, thresholds)

    return thresholds