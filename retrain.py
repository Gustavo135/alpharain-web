import firebase_admin
from firebase_admin import credentials, db
import tensorflow as tf
import numpy as np
import json
from datetime import datetime, timedelta

# --- 1. CONFIGURACIÓN DE FIREBASE ---
# IMPORTANTE: Descarga tu clave de servicio desde Firebase y guárdala como 'serviceAccountKey.json'
cred = credentials.Certificate('serviceAccountKey.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://alpharain-5a416-default-rtdb.firebaseio.com'
})

# --- 2. PARÁMETROS DEL MODELO (DEBEN SER IDÉNTICOS A LOS DE JS) ---
MIN_VALUES = np.array([20.0, 40.2, 13987.0, 1760.0])
MAX_VALUES = np.array([28.5, 62.0, 28422.0, 41930.0])
WEIGHTS_FILE = 'public/weights.json' # Ruta al archivo que usa la web

def create_model():
    """Crea la misma arquitectura de modelo que en TensorFlow.js"""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def normalize_data(data):
    """Normaliza los datos usando los mismos min/max que en el frontend."""
    return (data - MIN_VALUES) / (MAX_VALUES - MIN_VALUES)

# --- 3. LÓGICA DE RE-ENTRENAMIENTO ---
print("Iniciando script de re-entrenamiento...")

# Cargar datos de las últimas 24 horas desde Firebase
ref = db.reference('lecturas')
end_date = datetime.now()
start_date = end_date - timedelta(days=1)
query = ref.order_by_child('timestamp').start_at(start_date.isoformat()).end_at(end_date.isoformat())
all_data = query.get()

if not all_data:
    print("No se encontraron nuevos datos en las últimas 24 horas. Saliendo.")
    exit()

print(f"Se encontraron {len(all_data)} registros para re-entrenar.")

# Preparar datos para el entrenamiento (X: entradas, y: salidas)
# Asumimos que queremos predecir la temperatura de la siguiente hora
features = []
labels = []
# Convertir el diccionario a una lista ordenada por timestamp
sorted_data = sorted(all_data.values(), key=lambda x: x['timestamp'])

for i in range(len(sorted_data) - 1):
    current_reading = sorted_data[i]
    next_reading = sorted_data[i+1]
    
    input_data = [
        float(current_reading['temperatura']),
        float(current_reading['humedad']),
        int(current_reading['calidad_aire']),
        int(current_reading['luz'])
    ]
    output_data = float(next_reading['temperatura'])
    
    features.append(input_data)
    labels.append(output_data)

X_train = np.array(features)
y_train = np.array(labels)

# Normalizar los datos
X_train_scaled = normalize_data(X_train)
# Normalizar también la temperatura de salida
y_train_scaled = (y_train - MIN_VALUES[0]) / (MAX_VALUES[0] - MIN_VALUES[0])

# Crear el modelo y cargar los pesos actuales
model = create_model()
with open(WEIGHTS_FILE, 'r') as f:
    weights_json = json.load(f)
    # Keras espera los pesos en un formato ligeramente diferente (pesos, sesgos)
    # Este bucle los adapta
    keras_weights = [np.array(weights_json[i*2]) for i in range(len(weights_json)//2)]
    for i in range(len(weights_json)//2):
        keras_weights[i] = [np.array(weights_json[i*2]), np.array(weights_json[i*2+1])]
    
    # Aplanar la lista para set_weights
    flattened_weights = [item for sublist in keras_weights for item in sublist]
    model.set_weights(flattened_weights)

print("Pesos actuales cargados. Re-entrenando el modelo...")

# Re-entrenar el modelo con los nuevos datos (fine-tuning)
model.fit(X_train_scaled, y_train_scaled, epochs=10, batch_size=8, verbose=0)

print("Re-entrenamiento completado. Guardando nuevos pesos...")

# Guardar los nuevos pesos en el formato que espera TensorFlow.js
new_weights = model.get_weights()
weights_for_js = []
for layer_weights in new_weights:
    weights_for_js.append(layer_weights.tolist())

with open(WEIGHTS_FILE, 'w') as f:
    json.dump(weights_for_js, f)

print(f"Nuevos pesos guardados en '{WEIGHTS_FILE}'.")