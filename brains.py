import pickle
import tensorflow.keras.utils as utils
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers

train = utils.image_dataset_from_directory(
    "Training",
    label_mode = "categorical",
    batch_size = 32,
    image_size = (256, 256),
    seed = 37,
    validation_split = 0.3,
    subset = "training",
)

test = utils.image_dataset_from_directory(
    "Training",
    label_mode = "categorical",
    batch_size = 32,
    image_size = (256, 256),
    seed = 37,
    validation_split = 0.3,
    subset = "validation",
)

class Net():
    def __init__(self, input_shape):
        self.model = models.Sequential()
        # Input: 256 x 256 x 3
        # First layer is convolution with:
        # Frame/kernel: 13 x 13, Strides/Step size: 3, Filters/Depth: 8
        self.model.add(layers.Conv2D(8, 13, strides = 3, 
            activation = "relu", input_shape = input_shape))
        # Output: 82 x 82 x 8
        # Next layer is maxpool, Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 41 x 41 x 8
        # Next layer is one row of padding (top) and one column of padding (left):
        self.model.add(layers.ZeroPadding2D(padding = ((1,0), (1,0))))
        # Output: 42 x 42 x 8
        # Next layer is convolution with:
        # Frame/kernel: 3 x 3, Strides/Step size: 1, Filters/Depth: 8
        self.model.add(layers.Conv2D(8, 3, strides = 1, 
            activation = "relu"))
        # Output: 40 x 40 x 8
        # Next layer is maxpool, Frame: 2 x 2, Strides: 2
        self.model.add(layers.MaxPool2D(pool_size = 2))
        # Output: 20 x 20 x 8
        # Next layer is convolution with:
        # Frame/kernel: 3 x 3, Strides/Step size: 1, Filters/Depth: 8
        self.model.add(layers.Conv2D(8, 3, strides = 1, 
            activation = "relu"))
        # Output: 18 x 18 x 8
        # Now flatten
        self.model.add(layers.Flatten())
        # Output length: 2592
        self.model.add(layers.Dense(1024, activation = "relu"))
        self.model.add(layers.Dense(256, activation = "relu"))
        self.model.add(layers.Dense(64, activation = "relu"))
        self.model.add(layers.Dense(4, activation = "softmax"))
        self.loss = losses.CategoricalCrossentropy()
        self.optimizer = optimizers.SGD(learning_rate = 0.0001)
        self.model.compile(
            loss = self.loss,
            optimizer = self.optimizer,
            metrics = ["accuracy"],
        )
    def __str__(self):
        self.model.summary()
        return ""

    def save(self, filename):
        self.model.save(filename)

net = Net((256, 256, 3))
print(net)

net.model.fit(
    train, 
    batch_size = 32,
    epochs = 100,
    verbose = 2,
    validation_data = test,
    validation_batch_size = 32,
)

save_path = 'saves/faces_model_save_2023_03_10_100_epochs'
net.save(save_path)
with open(f'{save_path}/class_names.data', 'wb') as f:
    pickle.dump(train.class_names, f)