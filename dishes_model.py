from os import path

from keras import utils, Input, Model
from keras.src.applications.efficientnet import EfficientNetB0
from keras.src.layers import GlobalAveragePooling2D, BatchNormalization, Dropout, Dense
from matplotlib import pyplot as plt

# Some definitions
NUM_CLASSES = 6
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3
BATCH_SIZE = 64
RANDOM_SEED = 58239

# Define input shape
INPUT_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS)

# Show image samples
filenames = ["dataset/Dishes/American/American_309.jpg", "dataset/Dishes/Chinese/Chinese_751.jpg",
             "dataset/Dishes/European/European_101.jpg", "dataset/Dishes/Indian/Indian_823.jpg",
             "dataset/Dishes/Japanese/Japanese_111.jpg", "dataset/Dishes/Korean/Korean_100.jpg"]
for i, filename in enumerate(filenames):
    plt.subplot(2, 3, i + 1)
    plt.imshow(plt.imread(filename, format=None))
    plt.title(f"{path.basename(filename)}")
    plt.axis("off")

# Creating training dataset directly from directory
train_ds = utils.image_dataset_from_directory(
    directory="dataset/dishes/",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    subset="training",
    seed=RANDOM_SEED,
    validation_split=0.1)

# Creating evaluation dataset directly from directory
validation_ds = utils.image_dataset_from_directory(
    directory="dataset/dishes/",
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    subset="validation",
    seed=RANDOM_SEED,
    validation_split=0.1)

# Create input layer
inputs = Input(shape=INPUT_SHAPE)

# Use EfficientNetB0 as basic model
basic_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights="imagenet")

# Freeze the pretrained weights
basic_model.trainable = False

# Rebuild top layers
outputs = GlobalAveragePooling2D(name="avg_pool")(basic_model.output)
outputs = BatchNormalization()(outputs)
outputs = Dropout(0.2, name="top_dropout")(outputs)
outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(outputs)

# Merge to new model
model = Model(inputs, outputs)

# Print model summary
model.summary()

# Compile and run training
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

# Plot accuracy metric
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "validation"], loc="upper left")

# Save the model
model.save("dishes_model.keras")
