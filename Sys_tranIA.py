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