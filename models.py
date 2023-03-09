# -*- coding: utf-8 -*-

"""
RFML keras models Module
"""

from keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Conv2D,
    Conv3D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling2D,
    MaxPooling3D,
    ZeroPadding2D,
)
from keras.layers.core import (
    Activation,
    Dense,
    Dropout,
    Flatten,
    Reshape,
)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from dataset import HelicalDataset
from loss_plot import TrainingPlot
from keras.models import Model, load_model
import numpy as np


class RFMLModel:
    """docstring for ConvModel"""

    def __init__(self, config):

        self.config = config
        # self.dataset = Dataset(self.config["data_path"])
        self.build_model()
        self.compile_model()
        self.build_callbacks()

        if self.config["transfer_weights"]:

            self.load_weights(self.config["transfer_weights"])

    def build_model(self):
        """
        Desc:   Build Model using Keras Functional API
                handled by children classes
        """
        pass

    def build_callbacks(self):
        """
        keras callback function list builder
        """
        self.callbacks = [

            EarlyStopping(monitor="val_loss", patience=15,
                          verbose=1, mode="auto"),
            ModelCheckpoint(filepath=str(self.config["save_path"]), verbose=1,
                            save_best_only=True, mode="auto"),
            TensorBoard(log_dir="./graph", histogram_freq=0,
                        write_graph=True, write_images=True),
            TrainingPlot(self.config["save_path"][:-3]),
        ]

    def load_weights(self, weight_path):
        """
        Load weights of model
        """
        self.model.load(weight_path)

    def compile_model(self):

        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train_model(self, dataset):
        """
        Desc: Preprocess data and train model

        """

        self.model_history = self.model.fit(
            dataset.x_data,
            dataset.y_data,
            epochs=self.config["num_"],
            batch_size=self.config["batch_size"],
            validation_split=0.20,
            callbacks=self.callbacks,
            shuffle=True,
        )

    def plot_confusion_matrix(self, x, y):
        pred_y = self.model.predict(x, batch_size=20)
        cm = confusion_matrix(y, pred_y)
        print(cm)
        df_cm = pd.DataFrame(cm,
                             index=[str(i) for i in range(self.num_classes)],
                             columns=[str(i) for i in range(self.num_classes)])
        plt.figure(figsize=(4, 3))
        sn.heatmap(df_cm, annot=True)
        plt.savefig(self.config["save_path"][:-3] + '_cm.png')
        return classification_report(y, pred_y)


class Conv1DModel(RFMLModel):

    def build_model(self):
        """
        @OriginalAuthor : Debashri Roy
        Desc: model
        """
        drouput_rate = 0.2

        input_layer = Conv1D(
            filters=2048,
            kernel_size=4,
            strides=8,
            input_shape=(self.dataset.timesteps, self.dataset.data_dim),
            padding="same",
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform"
        )

        x = Dropout(drouput_rate)(input_layer)
        x = Conv1D(
            filters=1024,
            kernel_size=2,
            strides=4,
            padding="same",
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform")(x)

        x = Flatten()(x)

        x = Dense(
            512,
            activation="tanh",
            use_bias=True,
            kernel_initializer="glorot_uniform")(x)

        x = BatchNormalization(
            momentum=0.9,
            epsilon=1e-06,
            weights=None)(x)

        x = Dropout(drouput_rate)(x)

        out = Dense(self.config["num_classes"],
                    activation="softmax",
                    use_bias=True,
                    kernel_initializer="glorot_uniform")(x)

        self.model = Model(input_layer, out)

        self.model.summary()

    def get_activation_map(self, out_file):
        """plot activation maps to the given file
        Args:
            out_file (string): path  to save visualization
        Returns:
            None: Description
        """

        return None


class Conv2DModel(RFMLModel):

    def build_model(self, input_shape=128, num_classes=4):
        """
        @OriginalAuthor : Debashri Roy
        Desc: model
        """
        dr = 0.2
        input_shape = (input_shape, input_shape, 1)

        visible = Input(shape=input_shape)
        conv1 = Conv2D(32, kernel_size=4, activation='relu')(visible)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(16, kernel_size=4, activation='relu')(pool1)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        flat = Flatten()(pool2)
        hidden1 = Dense(10, activation='relu')(flat)
        output = Dense(num_classes, activation='softmax')(hidden1)
        model = Model(inputs=visible, outputs=output)
        # summarize layers
        print(model.summary())

        self.model = model

    def train_model(self, data_generator):
        self.model.fit_generator(data_generator, epochs=10)

    def get_activation_map(self, out_file):
        """plot activation maps to the given file
        Args:
            out_file (string): path  to save visualization
        Returns:
            None: Description
        """

        return None


class Conv3DModel(RFMLModel):
    def __init__(self, config, time_dim, raster_dim, num_classes):
        self.time_dim = time_dim
        self.raster_dim = raster_dim
        self.num_classes = num_classes
        super().__init__(config=config)

    def build_model(self):
        """
        Desc: 3D convolutional model
        """
        dr = 0.2
        input_shape = (self.time_dim, self.raster_dim, self.raster_dim, 1)
        input_layer = Input(input_shape)
        # convolutional layers
        conv_layer1 = Conv3D(filters=8, kernel_size=(
            7, 7, 7), activation='relu')(input_layer)
        conv_layer2 = Conv3D(filters=16, kernel_size=(
            5, 5, 5), activation='relu')(conv_layer1)

        # add max pooling to obtain the most imformatic features
        pooling_layer1 = MaxPooling3D(pool_size=(3, 3, 3))(conv_layer2)
        # pooling layers (3,3,3) for time_dim > 25
        conv_layer3 = Conv3D(filters=32, kernel_size=(
            5, 5, 5), activation='relu')(pooling_layer1)
        conv_layer4 = Conv3D(filters=64, kernel_size=(
            3, 3, 3), activation='relu')(conv_layer3)
        pooling_layer2 = MaxPooling3D(pool_size=(3, 3, 3))(conv_layer3)

        # perform batch normalization on the convolution outputs before feeding it to MLP architecture
        pooling_layer2 = BatchNormalization()(pooling_layer2)
        flatten_layer = Flatten()(pooling_layer2)

        # create an MLP architecture with dense layers : 4096 -> 512 -> 10
        # add dropouts to avoid overfitting / perform regularization
        dense_layer1 = Dense(units=100, activation='relu')(flatten_layer)
        dense_layer1 = Dropout(0.4)(dense_layer1)
        dense_layer2 = Dense(units=50, activation='relu')(dense_layer1)
        dense_layer2 = Dropout(0.4)(dense_layer2)
        output_layer = Dense(units=self.num_classes,
                             activation='softmax')(dense_layer2)

        # define the model with input layer and output layer
        model = Model(inputs=input_layer, outputs=output_layer)
        # summarize layers
        print(model.summary())

        self.model = model

    def train_model(self, helical_data):

        def unison_shuffled_copies(a, b):
            """
            shuffle a,b in unison and return shuffled a,b
            """
            assert len(a) == len(b)
            p = np.random.RandomState(seed=42).permutation(len(a))

            return a[p], b[p]

        x_data, y_data = unison_shuffled_copies(helical_data.x_data_helix,
                                                helical_data.y_data_helix)
        num_samples = y_data.shape
        percent_split = 0.25
        samples_split = int(num_samples * percent_split)
        history = self.model.fit(np.expand_dims(
            x_data[samples_split:], axis=-1),
            y_data[samples_split:],
            epochs=self.config['num_epochs'],
            batch_size=self.config['batch_size'],
            validation_data=[
            np.expand_dims(x_data[:samples_split], axis=-1),
            y_data[:samples_split]],
            callbacks=self.callbacks,
            shuffle=True)
        self.model = load_model(self.config["save_path"])
        report = self.plot_confusion_matrix(
            np.expand_dims(x_data[:samples_split], axis=-1),
            y_data[:samples_split])


        with open(config["save_path"][-3] + "_stats.txt", "w") as txt:
            txt.write("configuration:", config["save_path"][-3])
            txt.write("model summary:", self.model.summary())
            txt.write("history", history)
            txt.write("report", report)
        # plot cm


if __name__ == '__main__':
    # dg = DataGenerator("../data/data_4_radio.txt", batch_size=100)
    import json
    model_details = json.load(open("config.json"))['model_details']
    model = Conv3DModel(model_details,
                        model_details["time_window"],
                        model_details["img_size"],
                        model_details["num_class"])
    helical_data = HelicalDataset(model_details["data_path"],
                                  model_details["time_window"],
                                  model_details["num_class"],
                                  model_details["img_size"])
    model.train_model(helical_data)
    model.model.save(model.config['save_path'])
