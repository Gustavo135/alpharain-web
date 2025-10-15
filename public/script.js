// CONFIGURACIÓN DE FIREBASE 
// Configuración para conectar con Firebase
const firebaseConfig = {
    apiKey: "AIzaSy...",
    authDomain: "alpharain-5a416.firebaseapp.com",
    databaseURL: "https://alpharain-5a416-default-rtdb.firebaseio.com",
    projectId: "alpharain-5a416",
    storageBucket: "alpharain-5a416.appspot.com",
    messagingSenderId: "...",
    appId: "..."
};

// Inicializa Firebase y obtiene referencia a la base de datos
firebase.initializeApp(firebaseConfig);
const database = firebase.database();

// Valores mínimos y máximos para normalizar los datos de entrada
const minValues = [20.0, 40.2, 13987.0, 1760.0];
const maxValues = [28.5, 62.0, 28422.0, 41930.0];

// 4. ELEMENTOS DEL HTML 
// Obtiene referencias a los elementos del DOM para mostrar datos
const statusElem = document.getElementById('status');
const tempElem = document.getElementById('temp-actual');
const humElem = document.getElementById('hum-actual');
const aireElem = document.getElementById('aire-actual');
const luzElem = document.getElementById('luz-actual');
const predElem = document.getElementById('prediccion-temp');
const timeElem = document.getElementById('timestamp');

// LÓGICA PRINCIPAL 
// Función principal asincrónica
async function main() {
    // Indica que se está creando el modelo
    statusElem.textContent = 'Creando modelo de IA...';
    
    // Crea el modelo secuencial de TensorFlow.js
    const model = tf.sequential();
    model.add(tf.layers.dense({inputShape: [4], units: 16, activation: 'relu'}));
    model.add(tf.layers.dense({units: 8, activation: 'relu'}));
    model.add(tf.layers.dense({units: 1}));
    
    // Carga los pesos pre-entrenados al modelo
    const weightTensors = modelWeights.map(w => tf.tensor(w));
    model.setWeights(weightTensors);

    statusElem.textContent = 'Modelo creado. Esperando datos...';
    console.log('Modelo de IA creado y pesos cargados.');

    // Escucha la última lectura en la base de datos
    const lecturasRef = database.ref('lecturas').orderByKey().limitToLast(1);
    lecturasRef.on('value', (snapshot) => {
        if (!snapshot.exists()) return;
        const lastKey = Object.keys(snapshot.val())[0];
        const latestData = snapshot.val()[lastKey];
        
        // Actualiza la UI con los datos recibidos
        tempElem.textContent = `${latestData.temperatura} °C`;
        humElem.textContent = `${latestData.humedad} %`;
        aireElem.textContent = latestData.calidad_aire;
        luzElem.textContent = latestData.luz;
        timeElem.textContent = `Última lectura: ${latestData.timestamp}`;

        // Prepara los datos para la predicción
        const inputData = [
            parseFloat(latestData.temperatura),
            parseFloat(latestData.humedad),
            parseInt(latestData.calidad_aire),
            parseInt(latestData.luz)
        ];
        // Normaliza los datos
        const scaledData = inputData.map((value, i) => (value - minValues[i]) / (maxValues[i] - minValues[i]));
        const inputTensor = tf.tensor2d([scaledData]);
        // Realiza la predicción con el modelo
        const predictionTensor = model.predict(inputTensor);
        const predictionScaled = predictionTensor.dataSync()[0];
        // Desnormaliza el resultado para mostrarlo en °C
        const tempMin = minValues[0]; const tempMax = maxValues[0];
        const predictionReal = (predictionScaled * (tempMax - tempMin)) + tempMin;
        predElem.textContent = `${predictionReal.toFixed(1)} °C`;
        statusElem.textContent = 'Predicción actualizada.';
    });
}

// Ejecuta la función principal
main();
