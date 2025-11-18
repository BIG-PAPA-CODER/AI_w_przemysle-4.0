import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Ścieżki do danych
base_dir = r"D:\mgr_2sem\AI 4.0\data\archive\casting_data\casting_data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

# Parametry obrazu
IMG_SIZE = (512, 512)
BATCH_SIZE = 32

# --- GENERATORY DANYCH ---

# Normalizacja i augmentacja (tylko na zbiorze treningowym)
train_datagen = ImageDataGenerator(
    rescale=1./255,            # normalizacja pikseli 0–1
    validation_split=0.2,      # 20% danych jako walidacja
    rotation_range=10,         # lekkie obracanie
    zoom_range=0.1,            # lekkie przybliżenie
    width_shift_range=0.1,     # przesunięcie poziome
    height_shift_range=0.1     # przesunięcie pionowe
)

# Zbiór testowy bez augmentacji
test_datagen = ImageDataGenerator(rescale=1./255)

# --- TRAIN i VALIDATION ---
train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# --- TEST ---
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# --- Weryfikacja ---
print("\n✅ Dane wczytane poprawnie:")
print("Trening:", train_gen.samples, "obrazów")
print("Walidacja:", val_gen.samples, "obrazów")
print("Test:", test_gen.samples, "obrazów")

# --- Podgląd przykładowych zdjęć ---
import matplotlib.pyplot as plt
import numpy as np

x_batch, y_batch = next(train_gen)
plt.figure(figsize=(8, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(np.squeeze(x_batch[i]), cmap='gray')
    plt.title(f"Klasa: {'Defekt' if y_batch[i] == 0 else 'OK'}")
    plt.axis('off')
plt.tight_layout()
plt.show()
