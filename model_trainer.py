import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# --- Configuration --- 
DATA_DIR = './data1'
MODEL_SAVE_PATH = 'invisink_model.h5'
IMAGE_SIZE = (30, 30)
NUM_CLASSES = 16 # 0-9, +, -, *, /, (,)
TEST_SPLIT_SIZE = 0.2   
EPOCHS = 20
BATCH_SIZE = 32

# --- Data Loading and Preprocessing ---
def load_data(data_dir):
    """Loads images and labels from the data directory."""
    images = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    
    # Manually define the class mapping to ensure consistency
    class_map = {name: i for i, name in enumerate(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'x', 'slash', '(', ')'])}
    # Convert special characters in filenames for folder compatibility
    class_map_inv = {v: k for k, v in class_map.items()} 

    print("Loading data...")
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        class_index = class_map.get(class_name)
        if class_index is None:
            print(f"Warning: Class '{class_name}' not in class_map. Skipping.")
            continue
            
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            # Read image in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                # Normalize pixel values to be between 0 and 1
                image = image / 255.0
                images.append(image)
                labels.append(class_index)
                
    print(f"Loaded {len(images)} images.")
    return np.array(images), np.array(labels), class_map

images, labels, class_map = load_data(DATA_DIR)
print(f"Class mapping: {class_map}")

# Reshape images to have a single channel (for grayscale)
images = images.reshape(-1, IMAGE_SIZE[0], IMAGE_SIZE[1], 1)

# One-hot encode the labels
labels_categorical = to_categorical(labels, num_classes=NUM_CLASSES)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels_categorical, test_size=TEST_SPLIT_SIZE, random_state=42, stratify=labels_categorical
)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# --- CNN Model Architecture ---
def create_model():
    """Creates and compiles the CNN model."""
    model = Sequential([
        # First convolutional layer
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling2D((2, 2)),
        
        # Second convolutional layer
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten the feature maps to feed into the dense layers
        Flatten(),
        
        # Dense layer
        Dense(128, activation='relu'),
        Dropout(0.5), # Dropout for regularization
        
        # Output layer
        Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model = create_model()
model.summary()

# --- Model Training ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test)
)
print("--- Model Training Finished ---")

# --- Evaluation and Visualization ---
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()

# --- Save the Model ---
model.save(MODEL_SAVE_PATH)
print(f"\nModel saved successfully to {MODEL_SAVE_PATH}")
