import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- Generación de un Dataset de Ejemplo (en caso de no existir fuentes de datos) ---
# Este dataset simula información sobre rutas y sus características,
# que podrían influir en el tiempo de viaje.

data = {
    'origen': ['Terminal paso del comercio', 'los Alcázares', 'Calima', 'Chiminangos', 'Flora Industrial',
               'Salomia', 'Popular', 'Manzanares', 'Fátima', 'Rio Cali', 'Piloto',
               'Terminal paso del comercio', 'los Alcázares', 'Calima', 'Chiminangos', 'Flora Industrial',
               'Salomia', 'Popular', 'Manzanares', 'Fátima', 'Rio Cali', 'Piloto'],
    'destino': ['los Alcázares', 'Calima', 'Flora Industrial', 'Popular', 'Rio Cali',
                'Manzanares', 'Fátima', 'Piloto', 'Rio Cali', 'San Nicolás', 'San Nicolás',
                'Salomia', 'Chiminangos', None, None, None,
                'Piloto', None, 'San Nicolás', None, None, None],
    'distancia_km': [5, 5, 8, 5, 10, 8, 12, 8, 9, 7, 10, 10, 6, 3, 2, 6, 5, 4, 9, 3, 5, 7],
    'hora_inicio': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 7, 8, 10, 11, 13, 14, 16, 17, 9, 11, 13],
    'dia_semana': ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo',
                   'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo',
                   'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo', 'Lunes'],
    'condicion_clima': ['Soleado', 'Nublado', 'Lluvioso', 'Soleado', 'Nublado', 'Lluvioso', 'Soleado',
                       'Nublado', 'Lluvioso', 'Soleado', 'Nublado', 'Lluvioso', 'Soleado', 'Nublado',
                       'Lluvioso', 'Soleado', 'Nublado', 'Lluvioso', 'Soleado', 'Nublado', 'Lluvioso', 'Soleado'],
    'tiempo_estimado_minutos': [10, 12, 15, 10, 20, 16, 25, 18, 22, 14, 20, 18, 11, 7, 5, 13, 9, 8, 19, 6, 10, 15] # Variable objetivo
}

df = pd.DataFrame(data)

# Preprocesamiento de datos
# Convertir variables categóricas a numéricas usando one-hot encoding
df = pd.get_dummies(df, columns=['origen', 'destino', 'dia_semana', 'condicion_clima'], dummy_na=True)

# Eliminar filas con valores nulos introducidos por el one-hot encoding de 'destino'
df = df.dropna(subset=[col for col in df.columns if 'destino_' in col])

# Separar características (X) y variable objetivo (y)
X = df.drop('tiempo_estimado_minutos', axis=1)
y = df['tiempo_estimado_minutos']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Desarrollo del Modelo de Aprendizaje Supervisado (Regresión) ---
# Se utilizará un modelo de Random Forest Regressor para predecir el tiempo de viaje.

# Inicializar el modelo
modelo_rf = RandomForestRegressor(n_estimators=100, random_state=42) # n_estimators: número de árboles en el bosque

# Entrenar el modelo
modelo_rf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo_rf.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f"Error Cuadrático Medio del modelo: {mse:.2f}")

# --- Uso del Modelo para Predecir el Tiempo de una Nueva Ruta ---
# Para predecir el tiempo de una nueva ruta, se deben proporcionar las características
# en el mismo formato que se usó para entrenar el modelo.

def predecir_tiempo_ruta(modelo, origen, destino, distancia, hora, dia, clima, all_columns):
    """
    Predice el tiempo de viaje para una nueva ruta utilizando el modelo entrenado.

    Args:
        modelo: El modelo de aprendizaje automático entrenado.
        origen (str): El origen de la ruta.
        destino (str): El destino de la ruta.
        distancia (float): La distancia de la ruta en kilómetros.
        hora (int): La hora de inicio del viaje.
        dia (str): El día de la semana del viaje.
        clima (str): La condición climática durante el viaje.
        all_columns (list): Todas las columnas esperadas por el modelo (para asegurar el orden correcto).

    Returns:
        float: El tiempo de viaje estimado en minutos.
    """
    nueva_ruta = pd.DataFrame({
        'distancia_km': [distancia],
        'hora_inicio': [hora],
        'origen_' + origen: [1],
        'destino_' + destino: [1],
        'dia_semana_' + dia: [1],
        'condicion_clima_' + clima: [1]
    })

    # Asegurar que la nueva instancia tenga todas las columnas necesarias
    for col in all_columns:
        if col not in nueva_ruta.columns:
            nueva_ruta[col] = 0

    # Seleccionar y ordenar las columnas en el mismo orden que durante el entrenamiento
    nueva_ruta = nueva_ruta[all_columns]

    tiempo_predicho = modelo.predict(nueva_ruta)
    return tiempo_predicho[0]