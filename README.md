# Data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.utils import class_weight

# 1. Load dataset
df = pd.read_csv('cervicalcancer.csv')
print(df.info())
print(df.head())

# 2. Ganti "?" dengan NaN dan tangani nilai NaN
df.replace("?", np.nan, inplace=True)
df.fillna(df.mean(numeric_only=True), inplace=True)
df.dropna(inplace=True)

# 3. Ubah kolom object menjadi numerik
for column in df.select_dtypes(include=['object']).columns:
    df[column] = LabelEncoder().fit_transform(df[column])

# 4. Analisis Korelasi dan Feature Selection
correlation_matrix = df.corr()
print("Korelasi dengan target (Biopsy):")
print(correlation_matrix['Biopsy'].sort_values(ascending=False))

# Visualisasi heatmap korelasi
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Heatmap Korelasi")
plt.show()

# Pilih fitur dengan korelasi signifikan
selected_features = correlation_matrix['Biopsy'][correlation_matrix['Biopsy'].abs() > 0.1].index.tolist()
selected_features.remove('Biopsy')
print("Fitur yang dipilih berdasarkan korelasi:", selected_features)

# Update dataset dengan fitur terpilih
df = df[selected_features + ['Biopsy']]

# 5. Pisahkan fitur (X) dan label (y)
X = df.drop('Biopsy', axis=1).values
y = df['Biopsy'].values

# 6. Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 7. Bagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Terapkan SMOTE pada data training
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)
print("Distribusi kelas setelah SMOTE:", Counter(y_train))

# 9. Random Forest untuk Feature Importance
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Visualisasi pentingnya fitur
feature_importance = rf_model.feature_importances_
sorted_idx = np.argsort(feature_importance)

plt.barh(np.array(selected_features)[sorted_idx], feature_importance[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance Menggunakan Random Forest")
plt.show()

# 10. Tambahkan dimensi untuk 1D CNN
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 11. Definisikan model CNN 1D
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Sigmoid untuk output biner
])

# Kompilasi model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hitung class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)

# 12. Pelatihan model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights
)

# Mulai cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Buat cross-validator
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inisialisasi daftar untuk menyimpan hasil akurasi tiap fold
cv_accuracies = []

for train_index, val_index in kfold.split(X, y):
    # Split data menjadi training dan validation set
    X_train_cv, X_val_cv = X[train_index], X[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]

    # Bentuk ulang data untuk CNN 1D
    X_train_cv = np.expand_dims(X_train_cv, axis=2)
    X_val_cv = np.expand_dims(X_val_cv, axis=2)

    # Latih ulang model dengan fold ini
    model_cv = Sequential([
        Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train_cv.shape[1], X_train_cv.shape[2])),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])

    model_cv.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_cv.fit(X_train_cv, y_train_cv, epochs=10, batch_size=32, verbose=0)

    # Evaluasi pada validation set
    y_pred_cv = (model_cv.predict(X_val_cv) > 0.5).astype(int)
    acc = accuracy_score(y_val_cv, y_pred_cv)
    cv_accuracies.append(acc)

# Cetak hasil cross-validation
print(f"Cross-Validation Accuracies: {cv_accuracies}")
print(f"Mean CV Accuracy: {np.mean(cv_accuracies):.2f}")

# 13. Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 14. Visualisasi hasil pelatihan
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
