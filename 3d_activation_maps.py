import glob
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras import models


def plot_activation_map(model_path, img_array):

    model = models.load_model(model_path)
    layer_outputs = [layer.output for layer in model.layers[1:7]]
    activation_model = models.Model(
        inputs=model.input,
        outputs=layer_outputs,
    )
    expanded_input = np.expand_dims(np.expand_dims(
        img_array, axis=-1), axis=0
    )
    activations = activation_model.predict(expanded_input)

from keras.models import load_model


def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                    keys.remove(key)


def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)


def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
        fig.canvas.draw()


def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])


def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])


def activation_model(model_path):
    model = load_model(model_path)
    print(model.summary())
    layer_outputs = [layer.output for layer in model.layers[1:7]]
    activation_model = models.Model(
        inputs=model.input,
        outputs=layer_outputs,
    )
    return activation_model


def get_activations(model, data):
    expanded_input = np.expand_dims(np.expand_dims(
                                    data, axis=-1), axis=0)
    return model.predict(expanded_input)

