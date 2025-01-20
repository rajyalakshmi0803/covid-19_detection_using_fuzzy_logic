# %% Import Libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import numpy as np
import cv2
import skfuzzy as fuzz  # Fuzzy logic library
from skfuzzy.control import Antecedent, Consequent, Rule, ControlSystem, ControlSystemSimulation
from tensorflow.keras.callbacks import LearningRateScheduler

# %% Data Loading and Preprocessing
data = []
labels = []
path1 = r"C:\Users\tanis\OneDrive\Desktop\capstone\covid_reduced_dataset\covid\images"
path2 = r"C:\Users\tanis\OneDrive\Desktop\capstone\covid_reduced_dataset\normal\images"

# Load COVID images
for i in os.listdir(path1):
    labels.append("covid")
    img = cv2.imread(os.path.join(path1, i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)

# Load normal images
for i in os.listdir(path2):
    labels.append("normal")
    img = cv2.imread(os.path.join(path2, i))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    data.append(img)

data = np.array(data, dtype=np.float32) / 255.0
labels = np.array(labels)

# One-hot encoding labels
lb = LabelBinarizer()
labels_num = lb.fit_transform(labels)
labels = to_categorical(labels_num)

# %% Train-Test Split
(xtrain, xtest, ytrain, ytest) = train_test_split(data, labels, test_size=0.3, stratify=labels_num, random_state=42)

# %% Data Augmentation
aug = ImageDataGenerator(rotation_range=10, fill_mode="nearest")

# %% CNN Model Creation (User-Defined Model)
baseModel = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
])

cnn_model = Sequential([
    baseModel,
    MaxPooling2D(pool_size=(4, 4)),
    Flatten(name="flatten"),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax')
])

# %% Learning Rate Scheduler
def scheduler(epoch, lr):
    return lr * (1 / (1 + 0.1 * epoch))

lr_scheduler = LearningRateScheduler(scheduler)

# %% CNN Model Compilation
epochs = 25
lr = 1e-3
BS = 8
adam = Adam(learning_rate=lr)
cnn_model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])

# %% Initialize Model with Dummy Input
dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
cnn_model.predict(dummy_input)  # Initialize the model

# %% CNN Training
history = cnn_model.fit(
    aug.flow(xtrain, ytrain, batch_size=BS),
    steps_per_epoch=len(xtrain) // BS,
    validation_data=(xtest, ytest),
    validation_steps=len(xtest) // BS,
    epochs=epochs,
    callbacks=[lr_scheduler]
)

# %% Feature Extraction
# Correctly define the feature extractor with the model inputs and outputs
feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("flatten").output)

# Extract features from training and test sets
train_features = feature_extractor.predict(xtrain)
test_features = feature_extractor.predict(xtest)

# %% Fuzzy System Implementation
intensity = Antecedent(np.linspace(0, 1, 100), 'intensity')
sharpness = Antecedent(np.linspace(0, 1, 100), 'sharpness')
output = Consequent(np.linspace(0, 1, 100), 'classification')

intensity['low'] = fuzz.trimf(intensity.universe, [0, 0, 0.5])
intensity['medium'] = fuzz.trimf(intensity.universe, [0.25, 0.5, 0.75])
intensity['high'] = fuzz.trimf(intensity.universe, [0.5, 1, 1])

sharpness['low'] = fuzz.trimf(sharpness.universe, [0, 0, 0.5])
sharpness['medium'] = fuzz.trimf(sharpness.universe, [0.25, 0.5, 0.75])
sharpness['high'] = fuzz.trimf(sharpness.universe, [0.5, 1, 1])

output['normal'] = fuzz.trimf(output.universe, [0, 0, 0.5])
output['covid'] = fuzz.trimf(output.universe, [0.5, 1, 1])

rule1 = Rule(intensity['high'] & sharpness['high'], output['covid'])
rule2 = Rule(intensity['low'] | sharpness['low'], output['normal'])
rule3 = Rule(intensity['medium'] & sharpness['medium'], output['normal'])

fuzzy_ctrl = ControlSystem([rule1, rule2, rule3])
fuzzy_sim = ControlSystemSimulation(fuzzy_ctrl)

# %% Applying Fuzzy Logic to CNN Features
predictions = []
for i in range(test_features.shape[0]):
    feature = test_features[i]
    fuzzy_sim.input['intensity'] = np.mean(feature)
    fuzzy_sim.input['sharpness'] = np.std(feature)
    fuzzy_sim.compute()
    predictions.append(fuzzy_sim.output['classification'])

predictions = np.array(predictions)
predictions = (predictions > 0.5).astype(int)

# %% Classification Report
print(classification_report(ytest.argmax(axis=1), predictions, target_names=lb.classes_))
