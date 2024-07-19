### Don't forget to check installation of split-folders

import os
from warnings import filterwarnings
import pathlib
import keras
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import pandas as pd
import seaborn as sns
import splitfolders
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
filterwarnings('ignore')

df_path = '/kaggle/input/rice-image-dataset/Rice_Image_Dataset'

images = []
labels = []
for subfolder in os.listdir(df_path):
    
    subfolder_path = os.path.join(df_path, subfolder)
    if not os.path.isdir(subfolder_path):
        continue
  

    for image_filename in os.listdir(subfolder_path):

        image_path = os.path.join(subfolder_path, image_filename)
        images.append(image_path)
    
        labels.append(subfolder)
 
df = pd.DataFrame({'image': images, 'label': labels})

sns.set(style="whitegrid", palette="pastel")
ax = sns.countplot(x=df.label, color="gray")
ax.set_xlabel("Classes", fontdict={'weight': 'bold'})
ax.xaxis.label.set_color('black')
ax.set_ylabel("Number of samples", fontdict={'weight': 'bold'})
ax.yaxis.label.set_color('black')
plt.show()

fig, axs = plt.subplots(5, 4, figsize=(15, 15))

categories = df['label'].unique()
num_categories = len(categories)
for i, category in enumerate(categories):
   
    filepaths = df[df['label'] == category]['image'].values[:4]

    for j, filepath in enumerate(filepaths):
        ax = axs[i, j]
        image = imread(filepath)
        ax.imshow(image)
        ax.axis('off')

    ax.text(300, 100, category, fontdict={'weight': 'bold'}, fontsize=25, color='black')

plt.tight_layout()
plt.show()

df_path = pathlib.Path(df_path)
splitfolders.ratio(df_path, output='df_splitted', seed=42, ratio=(0.7, 0.15, 0.15))

BATCH_SIZE = 32
IMAGE_SIZE = (250, 250)
Train = keras.utils.image_dataset_from_directory(
    directory='/kaggle/working/df_splitted/train',
    labels='inferred',
    label_mode='categorical',
    batch_size= BATCH_SIZE,
    image_size= IMAGE_SIZE,
    seed= 42
)

Test = keras.utils.image_dataset_from_directory(
    directory= '/kaggle/working/df_splitted/test',
    labels='inferred',
    label_mode= 'categorical',
    batch_size= BATCH_SIZE,
    image_size= IMAGE_SIZE,
    seed= 42
)

Validation = keras.utils.image_dataset_from_directory(
    directory= '/kaggle/working/df_splitted/val',
    labels= 'inferred',
    label_mode= 'categorical',
    batch_size= BATCH_SIZE,
    image_size= IMAGE_SIZE,
    seed= 42
)

train_batch = next(iter(Train))
validation_batch = next(iter(Validation))
test_batch = next(iter(Test))
image_batch, labels_batch = train_batch
print(f"Train Shape: {image_batch.shape} (Batches = {len(Train)})")
print(f"Train label: {labels_batch.shape}\n")
image_batch, labels_batch = validation_batch
print(f"Validation Shape: {image_batch.shape} (Batches = {len(Validation)})")
print(f"Validation label: {labels_batch.shape}\n")
image_batch, labels_batch = test_batch
print(f"Test Shape: {image_batch.shape} (Batches = {len(Test)})")
print(f"Test label: {labels_batch.shape}\n")

INPUT_SHAPE = (250, 250, 3)

cnn = Sequential()
cnn.add(Conv2D(6, (3, 3), activation='relu', input_shape=INPUT_SHAPE))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Conv2D(16, (3, 3), activation='relu'))
cnn.add(MaxPooling2D((2, 2)))
cnn.add(Flatten())
cnn.add(Dense(120, activation='relu'))
cnn.add(Dense(84, activation='relu'))
cnn.add(Dense(5, activation='softmax'))
cnn.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
cnn.summary()
CNN_model = cnn.fit(Train, validation_data= Validation, epochs=20,
                    callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, mode='auto'))

training_loss = CNN_model.history['loss']
val_loss = CNN_model.history['val_loss']
training_acc = CNN_model.history['accuracy']
val_acc  = CNN_model.history['val_accuracy']

epoch_count = np.arange(1, len(training_loss) + 1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch_count, training_loss, 'r--', color='gray', label='Training Loss')
ax.plot(epoch_count, val_loss, '--bo', color='purple', linewidth=2.5, label='Validation Loss')
ax.legend()
ax.set_title('Number of epochs & Loss of CNN Model')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_xticks(np.arange(1, 16))
ax.grid(False)
plt.show()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epoch_count, training_acc, 'r--', color='gray', label='Training Accuracy')
ax.plot(epoch_count, val_acc, '--bo', color='purple', label='Validation Accuracy')
ax.legend()
ax.set_title('Number of epochs and Accuracy of CNN Model')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_xticks(np.arange(1, 16))
ax.grid(False)
plt.show()

cnn.save('model.h5')
last_model = tf.keras.models.load_model('model.h5')
last_model.summary()

test_loss, test_acc = last_model.evaluate(Test)
print(f'\nTest accuracy:{test_acc:.5f}\ntest_loss: {test_loss:.5f}')

predictions = []
true_labels = []
class_names = [
    "Arborio",
    "Basmati",
    "Ipsala",
    "Jasmine",
    "Karacadag"
    ]

for batch in Test:
    images, labels = batch
    batch_predictions = cnn.predict(images, verbose=0)
    predictions.extend(np.argmax(batch_predictions, axis=1))
    true_labels.extend(np.argmax(labels, axis=1))

predictions = np.array(predictions)
true_labels = np.array(true_labels)

conf_matrix = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='RdGy')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

class_report = classification_report(true_labels, predictions, target_names=class_names)
print("\nClassification Report:\n", class_report)