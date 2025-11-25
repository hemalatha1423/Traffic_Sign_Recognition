import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

# -------------------------------
# Step 1: Load and preprocess data
# -------------------------------
data = []
labels = []
classes = 43
cur_path = os.getcwd()

print("🔹 Loading training images...")

for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)
    for img_name in images:
        try:
            image = Image.open(path + '\\' + img_name)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("⚠️ Error loading image:", img_name)

data = np.array(data)
labels = np.array(labels)
print("✅ Data Loaded Successfully!")
print("Data Shape:", data.shape)
print("Labels Shape:", labels.shape)

# Normalize data
data = data / 255.0

# -------------------------------
# Step 2: Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

y_train = to_categorical(y_train, classes)
y_test = to_categorical(y_test, classes)

# -------------------------------
# Step 3: Build CNN model
# -------------------------------
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(32, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# -------------------------------
# Step 4: Train the model
# -------------------------------
epochs = 15
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=epochs,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Step 5: Save the model
# -------------------------------
model.save("traffic_classifier.h5")
print("💾 Model saved as traffic_classifier.h5")

# -------------------------------
# Step 6: Plot & Save Accuracy/Loss Graphs
# -------------------------------
plt.figure(figsize=(10,4))

# Accuracy
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')

plt.tight_layout()
plt.savefig("accuracy_loss_graph.png")
plt.show()
print("📊 Saved: accuracy_loss_graph.png")

# -------------------------------
# Step 7: Evaluate Model Performance
# -------------------------------
print("\n🔹 Evaluating Model...")

# Predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = np.argmax(y_test, axis=1)

# Metrics
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

print("\n📊 Evaluation Metrics:")
print(f"Accuracy : {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall   : {rec*100:.2f}%")
print(f"F1 Score : {f1*100:.2f}%")

# -------------------------------
# Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.savefig("confusion_matrix.png")
plt.show()

print("📈 Saved: confusion_matrix.png")
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.savefig("History.png")
plt.show()
print("📈 History.png")