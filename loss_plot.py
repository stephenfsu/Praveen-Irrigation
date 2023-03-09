# -*- coding: utf-8 -*-

import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
import keras
import numpy as np

class TrainingPlot(keras.callbacks.Callback):

    def __init__(self, save_path, *args, **kwargs):
        self.save_path = save_path
        super(self).__init__(*args, **kwargs)

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        # if len(self.losses) > 0:

        N = np.arange(0, len(self.losses))

        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        plt.style.use("seaborn")

        plt.figure()
        plt.plot(N, self.losses, label="train_loss")
        plt.plot(N, self.val_losses, label="val_loss")
        plt.title("Training and Validation Loss".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(self.save_path + "_loss.png".format(epoch))
        plt.close()

        plt.figure()
        plt.plot(N, self.acc, label="train_acc")
        plt.plot(N, self.val_acc, label="val_acc")
        plt.title("Training and Validation Accuracy".format(epoch))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig(self.save_path + "_acc.png".format(epoch))
        plt.close()
