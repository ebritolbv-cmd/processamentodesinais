import numpy as np
import os
import librosa
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import kagglehub

# 1. Configurações e Caminhos
classes_names = [
    "chainsaw", "crackling_fire", "dog", "rain", "sea_waves",
    "clock_tick", "crying_baby", "helicopter", "rooster", "sneezing"
]
Fs = 44100  # Taxa de amostragem padrão
n_mfcc = 40 # Aumentando para capturar mais detalhes para a CNN
max_len = 431 # Aproximadamente 5 segundos de áudio com hop_length=512

# Download do dataset
path = kagglehub.dataset_download("sreyareddy15/esc10rearranged")
base_path = os.path.join(path, "Data")

def extract_mfcc_2d(file_path, n_mfcc=40, max_len=431):
    try:
        audio, sr = librosa.load(file_path, sr=Fs)
        # Pad ou truncar o áudio para ter um tamanho fixo (5 segundos)
        target_length = 5 * Fs
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
            
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Garantir que o shape seja fixo (n_mfcc, max_len)
        if mfcc.shape[1] < max_len:
            mfcc = np.pad(mfcc, ((0, 0), (0, max_len - mfcc.shape[1])))
        else:
            mfcc = mfcc[:, :max_len]
            
        return mfcc
    except Exception as e:
        print(f"Erro ao processar {file_path}: {e}")
        return None

# 2. Carregamento dos Dados
X = []
y = []

print("Iniciando extração de características...")
for label in classes_names:
    folder_path = os.path.join(base_path, label)
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    print(f"Processando classe: {label} ({len(files)} arquivos)")
    for f in files:
        file_path = os.path.join(folder_path, f)
        mfcc = extract_mfcc_2d(file_path, n_mfcc=n_mfcc, max_len=max_len)
        if mfcc is not None:
            X.append(mfcc)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Adicionar dimensão de canal para a CNN (n_samples, n_mfcc, time, 1)
X = X[..., np.newaxis]

# Codificar labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(classes_names)

# Split de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

print(f"Shape de X_train: {X_train.shape}")

# 3. Construção do Modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(n_mfcc, max_len, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Treinamento
print("Iniciando treinamento...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# 5. Avaliação Final
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nAcurácia no conjunto de teste: {test_acc:.4f}")

# Salvar resultados para o relatório
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia do Modelo')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda do Modelo')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()

plt.tight_layout()
plt.savefig('/home/ubuntu/resultado_cnn.png')
print("Gráfico de desempenho salvo em /home/ubuntu/resultado_cnn.png")
