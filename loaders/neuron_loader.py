"""
Original version by Mu and Andreas, https://arxiv.org/abs/2006.14032 (licenced under CC-BY-SA)

Loads the activations for all 512 neurons (from the final hidden layer of resnet) using the standard forward() combined with the
register_forward_hook() (see definition of hook_function)
"""

import os
import numpy as np
import torch
import tqdm
import imageio

import S
import loaders.metadata_loader as MetaDataLoader
import settings

segmentation_data = None
mean = 0
neuron_activations_batch = np.array([])

MEAN = [109.5388, 118.6897, 124.6901]


def hook_function(module, inp, output):
    """
    In model_loader.py, there is a line

                hook_model.register_forward_hook(hook_fn)

    Executing that line makes it so that this function (which is the hook_fn) referenced there is executed (once) during a forward pass of
     the neural network. This happens during the execution of

                with torch.no_grad():
                    logits = model.forward(input_)

    At that point, the respective batch of the data we need is put into neuron_activations_batch.
    """
    global neuron_activations_batch
    neuron_activations_batch = output.data.cpu().numpy()

def init():
    global mean
    mean = [109.5388, 118.6897, 124.6901]

def get_neuron_values(model):
    def compute_neuron_values():
        def normalize_image(rgb_img):
            img = np.array(rgb_img, dtype=np.float32)
            img = img[:, :, ::-1]

            img -= MEAN
            img = img.transpose((2, 0, 1))
            return img

        filename_neuron_activations_temp = "results/inspected_neurons_temp.mmap"

        n_batches = S.n_images // settings.TALLY_BATCH_SIZE
        if n_batches * settings.TALLY_BATCH_SIZE < S.n_images:
            n_batches += 1

        for batch_i in tqdm.tqdm(range(n_batches), desc="Extracting neuron activations"):
            L, R = batch_i * settings.TALLY_BATCH_SIZE, min((batch_i + 1) * settings.TALLY_BATCH_SIZE, S.n_images)
            image_filenames = MetaDataLoader.map_image_i_2_image_filename[L: R]

            images = [imageio.imread(os.path.join(settings.IMAGE_DIRECTORY, filename)) for filename in image_filenames]
            images = [normalize_image(image) for image in images]
            images = np.array(images)

            # ------------------------------------------------------------------------------------------------------------------------
            # - This is a remnant from the previous implementation. I don't know why it's done this way, but I want to keep the same -
            # -   format to obtain identical neuron activations.                                                                     -
            # ------------------------------------------------------------------------------------------------------------------------
            input_ = torch.from_numpy(images[:, ::-1, :, :].copy())
            input_.div_(255 * 0.224)
            if settings.GPU:
                input_ = input_.cuda()


            with torch.no_grad():
                logits = model.forward(input_)


            if batch_i == 0:  # now that we can infer the shape, initialize result arrays
                print(f"Batch size is {settings.TALLY_BATCH_SIZE}; Logits shape = {logits.shape}, Neuron Activations shape = {np.shape(neuron_activations_batch)}")
                SH = (S.n_images, *neuron_activations_batch.shape[1:])
                np.save(filename_shape, SH)
                map_im_n_2_activations = np.memmap(filename_neuron_activations_temp, dtype=np.float32, mode="w+", shape=SH)

            map_im_n_2_activations[L: R] = neuron_activations_batch

        print("Reordering neuron activations to neuron -> image -> activations (from image -> neuron -> activations)")
        map_n_im_2_activations = np.memmap(filename_neuron_activations, dtype=np.float32, mode="w+", shape=(SH[1], SH[0], SH[2], SH[3]))
        map_n_im_2_activations[:] = np.memmap.swapaxes(map_im_n_2_activations, 0, 1)[:]

        print("Deleting the old (image -> neuron -> activations) file")
        map_im_n_2_activations._mmap.close()
        del map_im_n_2_activations
        os.remove(filename_neuron_activations_temp)

    # ------------------------------------------------------------------------------------------------------
    # - In one experiment, going from a memmap to a regular array has changed the runtime from 264 to 260. -
    # - Since a memmap is also more memory efficient, this makes it altogether a better choice.            -
    # ------------------------------------------------------------------------------------------------------
    filename_shape = "results/inspected_neurons_shape.npy"
    filename_neuron_activations = "results/inspected_neurons.mmap"

    if not os.path.exists(filename_shape) or not os.path.exists(filename_neuron_activations):
        compute_neuron_values()

    print("Returning neuron values from disk")
    neuron_activations_shape = np.load(filename_shape)
    S.n_images = neuron_activations_shape[0]
    S.n_neurons = neuron_activations_shape[1]
    S.mask_shape = (neuron_activations_shape[2], neuron_activations_shape[3])
    S.n_pixels = S.n_images * S.mask_shape[0] * S.mask_shape[1]

    return np.memmap(filename_neuron_activations, dtype=np.float32, mode="c", shape=(S.n_neurons, S.n_images, *S.mask_shape))