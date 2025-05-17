import openml
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from betacal import BetaCalibration
from netcal.binning import BBQ
from venn_abers import VennAbersCalibrator
from sklearn.preprocessing import StandardScaler
import calibration as cal

output_dir = "calibration_results_c5_md"
os.makedirs(output_dir, exist_ok=True)

datasets_ids = [310, 725, 761, 803, 807, 819, 833, 847, 923, 959, 1019, 1046, 1053, 1471, 1489, 4154, 4534, 41526, 45023, 45036, 46281, 46298]

n_repeats = 10
n_splits = 10

def save_combined_calibration_plot(dataset_id, calibration_results, y_test):
    plt.figure(figsize=(10, 8))
    for method, prob in calibration_results.items():
        #y_test son las etiquetas verdaderas para el conjunto de prueba
        #prob es el array de probabilidades predichas por el modelo en el conjunto de prueba
        #n_bin = 10 especifica el número de bins en que se divide el rango de probabilidades[0,1]
        #Outputs
        #prob_true es un array de 10 elementos(float) que contiene la fracción de positivos en cada bin(la observación real de la clase positiva)
        #prob_pred es un array de 10 elementos(float) que contiene la media de las probabilidades predichas en cada bin
        prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)

        #grafica lo que predice el modelo "prob_pred" vs lo que realmente ocurre "prob_true"
        #el nombre de cada método de calibración se añade a la leyenda "label=method"
        plt.plot(prob_pred, prob_true, marker='o', label=method)
        #coordenadas de una curva de calibración perfecta, "k--" es el estilo de la línea (línea discontinua negra)"
        plt.plot([0,1],[0,1], "k--", label="Perfectamente calibrado")
        #etiquetas y título del gráfico
        plt.xlabel("Predicción media de la clase positiva por el modelo")
        plt.ylabel("Observaciones reales de la clase positiva")
        plt.title(f"Comparación de curvas de calibración (Dataset {dataset_id})")
        plt.legend(loc="best")
        #se guarda el gráfico en el directorio de salida como un PNG
        plt_file_rute = os.path.join(output_dir, f"calibration_curves_dataset_{dataset_id}.png")
        plt.savefig(plt_file_rute)
        plt.close()

for dataset_id in datasets_ids:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

    # Preprocesamiento
    #LabelEncoder convierte etiquetas categóricas en enteros
    le = LabelEncoder()
    y = le.fit_transform(y)
    #pd.DataFrame convierte la matriz X en un DataFrame de pandas
    X = pd.DataFrame(X)
    #pd.get_dummies convierte variables categóricas en variables dummy (0/1)
    X = pd.get_dummies(X)
    #SimpleImputer reemplaza los valores NaN por la media de cada columna
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    #StandardScaler normaliza los datos para que tengan media 0 y varianza 1
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Inicializar el diccionario usado para el almacenamiento de métricas
    all_metrics = {
        "Base Model": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []},
        "Platt Scaling": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []},
        "Isotonic Regression": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []},
        "Beta Calibration": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []},
        "BBQ": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []},
        "Venn-Abers": {"Accuracy": [], "Brier Score": [], "Log Loss": [], "ECE": []}
    }
    all_probabilities = {method: [] for method in all_metrics.keys()}
   
    y_true_all = []
    #repeticiones de validación cruzada
    for repetition in range(n_repeats):

        #StratifiedKFold divide el conjunto de datos en n_splits partes, manteniendo la proporción de clases
        #shuffle=True mezcla los datos antes de dividirlos para evitar sesgos
        #random_state=42 asegura que la división sea reproducible
        random_state_number = 42
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state_number)
        #skf.split(X,y) devuelve los índices de entrenamiento y test para cada división
        #para cada división de los datos, se entrena y evalúa el modelo
        for train_idx, test_idx, in skf.split(X, y):
            #Aquí se dividen los datos en conjuntos de entrenamiento y prueba
            #X_train contiene los datos de entrenamiento para la división actual (indicada por train_idx)
            #X_test contiene los datos de prueba para la división actual (indicada por test_idx)
            X_train, X_test = X[train_idx], X[test_idx]
            #y_train contiene las etiquetas de entrenamiento para la división actual (indicada por train_idx)
            #y_test contiene las etiquetas de prueba para la división actual (indicada por test_idx)
            y_train, y_test = y[train_idx], y[test_idx]
            #Ahora se acumulan todas las etiquetas verdaderas de prueba en una lista para su posterior uso
            #en el cálculo de métricas globales
            y_true_all.extend(y_test)

            #1. Primero, obtenemos subconjuntos de validación a partir de los conjuntos de entrenamiento
            #A training subset (X_train_split and y_train_split) that will be used to train a model.
            #A validation subset (X_val and y_val) that will be used to evaluate the model's performance during training.
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=random_state_number)

            #2. Entrenamos el modelo base con el conjunto de entrenamiento
            #DecisionTreeClassifier crea un clasificador de árbol de decisión
            #criterion='entropy' usa la entropía como criterio de división. Con entropía queremos decir que el árbol
            #de decisión se construye de tal manera que minimiza la entropía de la información en cada nodo, esto es,
            #  el árbol se construye de tal manera que la información ganada en cada división es máxima.
            #Entropía es una medida de la incertidumbre o impureza de un conjunto de datos.
            #random_state=42 asegura que el modelo sea reproducible
            #max_depth=20 limita la profundidad máxima del árbol a 20 niveles
            #min_samples_leaf=5 asegura que cada hoja del árbol tenga al menos 5 muestras
            clf = DecisionTreeClassifier(criterion='entropy', random_state=random_state_number, max_depth=20, min_samples_leaf=5)
            #se entrena el modelo con los datos de entrenamiento
            clf.fit(X_train_split, y_train_split)
            #predict_proba devuelve las probabilidades predichas para cada clase por el modelo

            #The slicing operation [:, 1] selects all rows (:) and the second column (1) of the 2D array returned by predict_proba.
            #The method predict_proba predicts the class probabilities for each sample in the input dataset X_test.
            #The output of predict_proba is a 2D NumPy array of shape (n_samples, n_classes), where:
            #n_samples is the number of samples in X_test.
            #n_classes is the number of classes in the classification problem.
            #This extracts the predicted probabilities for the positive class (class 1) for all samples in X_test. These probabilities 
            #represent the model's confidence that each sample belongs to the positive class.
            prob_test = clf.predict_proba(X_test)[:, 1]
            #pred_test convierte las probabilidades en etiquetas binarias (0 o 1) usando un umbral de 0.5
            #si prob_test es mayor que 0.5, se asigna 1 (positivo), de lo contrario, se asigna 0 (negativo)
            #astype(int) convierte el resultado a tipo entero
            pred_test = (prob_test > 0.5).astype(int)

            #3. Evaluación del modelo base
            #requiere los parámetros: y_true y y_pred
            all_metrics["Base Model"]["Accuracy"].append(accuracy_score(y_test, pred_test))
            #requiere los parámetros: etiquetas verdaderas y probabilidades predichas para cada clase
            all_metrics["Base Model"]["Brier Score"].append(brier_score_loss(y_test, prob_test))
            #requiere los parámetros: etiquetas verdaderas y probabilidades predichas para cada clase
            all_metrics["Base Model"]["Log Loss"].append(log_loss(y_test, prob_test))
            #requiere los parámetros: etiquetas verdaderas y probabilidades predichas para cada clase
            all_metrics["Base Model"]["ECE"].append(cal.get_calibration_error(prob_test, y_test))

            all_probabilities["Base Model"].extend(prob_test)

            #4. Calibración del modelo usando Platt Scaling
            #sigmoid usa la función sigmoide para calibrar las probabilidades, en concordancia con la técnica de Platt Scaling
            #prefit indica que el modelo ya ha sido ajustado y no necesita ser ajustado nuevamente
            platt_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
            #.fit ajusta el clasificador calibrado a los datos de validación
            #a diferencia de los datos de entrenamiento, los datos de validación se utilizan para ajustar el modelo calibrado
            platt_clf.fit(X_val, y_val)
            #predict_proba devuelve las probabilidades predichas para cada clase por el modelo calibrado
            prob_platt = platt_clf.predict_proba(X_test)[:, 1]
            #accuracy_score calcula la precisión del modelo calibrado comparando las etiquetas verdaderas y las predicciones del modelo
            all_metrics["Platt Scaling"]["Accuracy"].append(accuracy_score(y_test, (prob_platt > 0.5).astype(int))) 
            all_metrics["Platt Scaling"]["Brier Score"].append(brier_score_loss(y_test, prob_platt))
            all_metrics["Platt Scaling"]["Log Loss"].append(log_loss(y_test, prob_platt))
            all_metrics["Platt Scaling"]["ECE"].append(cal.get_calibration_error(prob_platt, y_test))
            all_probabilities["Platt Scaling"].extend(prob_platt)

            #5. Calibración del modelo usando Isotonic Regression
            isotonic_clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
            isotonic_clf.fit(X_val, y_val)
            prob_isotonic = isotonic_clf.predict_proba(X_test)[:, 1]
            all_metrics["Isotonic Regression"]["Accuracy"].append(accuracy_score(y_test, (prob_isotonic > 0.5).astype(int)))
            all_metrics["Isotonic Regression"]["Brier Score"].append(brier_score_loss(y_test, prob_isotonic))
            all_metrics["Isotonic Regression"]["Log Loss"].append(log_loss(y_test, prob_isotonic))
            all_metrics["Isotonic Regression"]["ECE"].append(cal.get_calibration_error(prob_isotonic, y_test))
            all_probabilities["Isotonic Regression"].extend(prob_isotonic)

            #6. Calibración usando Beta Calibration
            beta_clf = BetaCalibration()
            #clf.predict_proba(X_val)[:, 1]

            #Esto obtiene las probabilidades predichas por un clasificador clf sobre un conjunto de validación X_val.
            #predict_proba(X_val) devuelve una matriz de probabilidades con una columna por clase.
            #[:, 1] selecciona las probabilidades de la clase positiva (generalmente la clase con etiqueta 1).
            #Entonces eso da un vector con las probabilidades de que cada muestra en X_val pertenezca a la clase positiva.
            beta_clf.fit(clf.predict_proba(X_val)[:, 1], y_val)
            #The predict method takes these raw probabilities as input and adjusts them using the calibration model
            #The result of the predict method, stored in prob_beta, is an array of calibrated probabilities. These probabilities
            #are adjusted to better reflect the true likelihood of the positive class.
            prob_beta = beta_clf.predict(prob_test)
            all_metrics["Beta Calibration"]["Accuracy"].append(accuracy_score(y_test, (prob_beta > 0.5).astype(int)))   
            all_metrics["Beta Calibration"]["Brier Score"].append(brier_score_loss(y_test, prob_beta))
            all_metrics["Beta Calibration"]["Log Loss"].append(log_loss(y_test, prob_beta))
            all_metrics["Beta Calibration"]["ECE"].append(cal.get_calibration_error(prob_beta, y_test))
            all_probabilities["Beta Calibration"].extend(prob_beta)

            #7. Calibración usando BBQ
            bbq_clf = BBQ()
            bbq_clf.fit(clf.predict_proba(X_val)[:, 1], y_val)
            prob_bbq = bbq_clf.transform(prob_test)
            all_metrics["BBQ"]["Accuracy"].append(accuracy_score(y_test, (prob_bbq > 0.5).astype(int)))
            all_metrics["BBQ"]["Brier Score"].append(brier_score_loss(y_test, prob_bbq))
            all_metrics["BBQ"]["Log Loss"].append(log_loss(y_test, prob_bbq))
            all_metrics["BBQ"]["ECE"].append(cal.get_calibration_error(prob_bbq, y_test))
            all_probabilities["BBQ"].extend(prob_bbq)

            #8. Calibración usando Venn-Abers
            va_clf = VennAbersCalibrator(estimator=clf)
            va_clf.fit(X_val, y_val)
            intervals = va_clf.predict_proba(X_test)
            prob_va = np.array([np.mean(interval) for interval in intervals])
            all_metrics["Venn-Abers"]["Accuracy"].append(accuracy_score(y_test, (prob_va > 0.5).astype(int)))
            all_metrics["Venn-Abers"]["Brier Score"].append(brier_score_loss(y_test, prob_va))
            all_metrics["Venn-Abers"]["Log Loss"].append(log_loss(y_test, prob_va))
            all_metrics["Venn-Abers"]["ECE"].append(cal.get_calibration_error(prob_va, y_test))
            all_probabilities["Venn-Abers"].extend(prob_va)

            # Promediar métricas acumuladas
    
    final_metrics = {
        method: {metric: np.mean(all_metrics[method][metric]) for metric in all_metrics[method].keys()}
        for method in all_metrics.keys()
    }
    # Crear gráfico combinado con probabilidades acumuladas
    save_combined_calibration_plot(dataset_id, all_probabilities, y_true_all)
    # Guardar métricas en un archivo JSON
    metrics_file = os.path.join(output_dir, f"metrics_dataset_{dataset_id}_repeated.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=4)


                   



