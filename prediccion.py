import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Clase para generar datos y trabajar con ellos
class DataGenerator:
    def __init__(self):
        pass

    def generate_synthetic_data(self):
        # Definir horas de estudio y calificaciones
        X = np.array([2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7,
                      7.7, 5.9, 4.5, 3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4,
                      2.7, 4.8, 3.8, 6.9, 7.8]).reshape(-1, 1)  # Horas de estudio
        
        y = np.array([21, 65, 55, 88, 60, 40, 90, 70, 83, 52,
                      78, 72, 62, 57, 30, 85, 21, 45, 75, 80,
                      55, 65, 58, 74, 82])  # Calificaciones correspondientes
        return X, y

    def load_data_from_excel(self, file_path, x_column, y_column):
        data = pd.read_excel(file_path)
        X = data[[x_column]].values
        y = data[y_column].values
        return X, y

# Clase para el modelo de regresión lineal y las operaciones principales
class LinearRegressionModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    # Para ver en detalle
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Error Cuadrático Medio (MSE): {mse:.2f}')
        print(f'Error Absoluto Medio (MAE): {mae:.2f}')
        print(f'Coeficiente de Determinación (R^2): {r2:.2f}')

    def visualize_results(self, X_train, y_train):
        plt.scatter(X_train, y_train, color='blue', label='Datos de Entrenamiento')
        plt.plot(X_train, self.predict(X_train), color='red', label='Línea de Regresión')
        plt.xlabel('Horas de Estudio')
        plt.ylabel('Calificaciones')
        plt.title('Horas de Estudio vs Calificaciones')
        plt.legend()
        plt.show()

# Código principal para entrenar y evaluar el modelo
if __name__ == "__main__":
    # Paso 1: Generación o carga de datos
    data_generator = DataGenerator()

    # Generar datos sintéticos basados en datos específicos
    X, y = data_generator.generate_synthetic_data()

    # Paso 2: Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=53)

    # Paso 3: Crear y entrenar el modelo
    model = LinearRegressionModel()
    model.train(X_train, y_train)

    # Paso 4: Evaluar el modelo
    model.evaluate(X_test, y_test)

    # Paso 5: Visualizar resultados del entrenamiento
    model.visualize_results(X_train, y_train)

    # Paso 6: Hacer una predicción para un nuevo valor
    new_hours = [[4]]  # Horas de estudio para la nueva predicción
    predicted_grade = model.predict(new_hours)
    print(f'Predicción de calificación para {new_hours[0][0]} horas de estudio: {predicted_grade[0]:.2f}')
