import json

# from models import (
#     ConvModel,
#     Conv1DModel,
#     GRUModel,
#     NNModel,
#     RNNvModel
# )

# MODELS = {
#     "conv": ConvModel,
#     "conv1D": Conv1DModel,
#     "GRU": GRUModel,
#     "NN": NNModel,
#     "RNN": RNNvModel,
# }

from models import (
    Conv2DModel,
    Conv1DModel,
    Conv3DModel,
)

MODELS = {
    "conv": Conv2DModel,
    "conv1D": Conv1DModel,
    "conv3D": Conv3DModel,
}

class Trainer:
    """
    Desc:   Wrapper to train keras model using configuration from
                    config.json

    """

    def __init__(self, config):
        """
        config = {
                        "model_type" :  <type of model (str)>

                        }

        """
        self.config = config
        self.model_holder = MODELS[self.config["model_type"]](
            self.config["model_details"])

    def train(self):
        """
        Desc:   Train workflow

        """
        self.model_holder.train_model()


if __name__ == "__main__":

    trainer = Trainer(json.load(open("config.json")))
    trainer.train()
