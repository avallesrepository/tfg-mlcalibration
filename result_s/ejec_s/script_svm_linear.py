import openml
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from betacal import BetaCalibration
from netcal.binning import BBQ
from venn_abers import VennAbersCalibrator
import calibration as cal  # Para calcular ECE

# Crear directorio para guardar resultados y gráficos
output_dir = "calibration_results_svm_radial_s"
os.makedirs(output_dir, exist_ok=True)

# Carga de datasets
dataset_ids = [3, 15, 29, 31, 37, 38, 44, 316, 953, 958, 962, 978, 1067, 1462, 1462, 1464, 1487, 1494, 1510, 4134, 4134, 40701]

# Número de repeticiones y particiones
n_repeats = 10
n_splits = 5

# Función para guardar gráficos comparativos
def save_combined_calibration_plot(dataset_id, calibration_results, y_test):
    plt.figure(figsize=(10, 8))
    
    for method, prob in calibration_results.items():
        prob_true, prob_pred = calibration_curve(y_test, prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=method)
    
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curves Comparison (Dataset {dataset_id})")
    plt.legend(loc="best")
    
    plot_file = os.path.join(output_dir, f"calibration_curves_dataset_{dataset_id}.png")
    plt.savefig(plot_file)
    plt.close()

# Procesar cada dataset
for dataset_id in dataset_ids:
    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    
    # Preprocesamiento
    le = LabelEncoder() 
    y = le.fit_transform(y) #etiquetas categóricas -> enteros
    X = pd.DataFrame(X) #para poder usar get_dummies
    X = pd.get_dummies(X) # Codificación one-hot para datos categóricos
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X) #NaN -> media de la columna

    # Inicializar almacenamiento de métricas
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

    # Repeticiones de validación cruzada
    for repetition in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repetition)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            y_true_all.extend(y_test)

            # Dividir X_train y y_train en un conjunto de entrenamiento y un conjunto de validación
            X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42)

            # Entrenar el modelo base con el conjunto de entrenamiento
            clf = SVC(probability=True, kernel='rbf', random_state=42)
            clf.fit(X_train_split, y_train_split)
            prob_test = clf.predict_proba(X_test)[:, 1]
            pred_test = (prob_test > 0.5).astype(int)

            # Evaluación del modelo base
            all_metrics["Base Model"]["Accuracy"].append(accuracy_score(y_test, pred_test))
            all_metrics["Base Model"]["Brier Score"].append(brier_score_loss(y_test, prob_test))
            all_metrics["Base Model"]["Log Loss"].append(log_loss(y_test, prob_test))
            all_metrics["Base Model"]["ECE"].append(cal.get_calibration_error(prob_test, y_test))
            all_probabilities["Base Model"].extend(prob_test)

            # Calibración del modelo usando Platt Scaling
            platt_clf = CalibratedClassifierCV(estimator=clf, method="sigmoid", cv="prefit")
            platt_clf.fit(X_val, y_val)
            prob_platt = platt_clf.predict_proba(X_test)[:, 1]
            all_metrics["Platt Scaling"]["Accuracy"].append(accuracy_score(y_test, (prob_platt > 0.5).astype(int)))
            all_metrics["Platt Scaling"]["Brier Score"].append(brier_score_loss(y_test, prob_platt))
            all_metrics["Platt Scaling"]["Log Loss"].append(log_loss(y_test, prob_platt))
            all_metrics["Platt Scaling"]["ECE"].append(cal.get_calibration_error(prob_platt, y_test))
            all_probabilities["Platt Scaling"].extend(prob_platt)

            # Calibración del modelo usando Isotonic Regression
            isotonic_clf = CalibratedClassifierCV(estimator=clf, method="isotonic", cv="prefit")
            isotonic_clf.fit(X_val, y_val)
            prob_isotonic = isotonic_clf.predict_proba(X_test)[:, 1]
            all_metrics["Isotonic Regression"]["Accuracy"].append(accuracy_score(y_test, (prob_isotonic > 0.5).astype(int)))
            all_metrics["Isotonic Regression"]["Brier Score"].append(brier_score_loss(y_test, prob_isotonic))
            all_metrics["Isotonic Regression"]["Log Loss"].append(log_loss(y_test, prob_isotonic))
            all_metrics["Isotonic Regression"]["ECE"].append(cal.get_calibration_error(prob_isotonic, y_test))
            all_probabilities["Isotonic Regression"].extend(prob_isotonic)

            # Calibración usando Beta Calibration
            beta_clf = BetaCalibration()
            beta_clf.fit(clf.predict_proba(X_val)[:, 1], y_val)
            prob_beta = beta_clf.predict(prob_test)
            all_metrics["Beta Calibration"]["Accuracy"].append(accuracy_score(y_test, (prob_beta > 0.5).astype(int)))
            all_metrics["Beta Calibration"]["Brier Score"].append(brier_score_loss(y_test, prob_beta))
            all_metrics["Beta Calibration"]["Log Loss"].append(log_loss(y_test, prob_beta))
            all_metrics["Beta Calibration"]["ECE"].append(cal.get_calibration_error(prob_beta, y_test))
            all_probabilities["Beta Calibration"].extend(prob_beta)

            # Calibración usando BBQ
            bbq_clf = BBQ()
            bbq_clf.fit(clf.predict_proba(X_val)[:, 1], y_val)
            prob_bbq = bbq_clf.transform(prob_test)
            all_metrics["BBQ"]["Accuracy"].append(accuracy_score(y_test, (prob_bbq > 0.5).astype(int)))
            all_metrics["BBQ"]["Brier Score"].append(brier_score_loss(y_test, prob_bbq))
            all_metrics["BBQ"]["Log Loss"].append(log_loss(y_test, prob_bbq))
            all_metrics["BBQ"]["ECE"].append(cal.get_calibration_error(prob_bbq, y_test))
            all_probabilities["BBQ"].extend(prob_bbq)

            # Calibración usando Venn-Abers
            va_clf = VennAbersCalibrator(estimator=clf)

            # Convertir X_val y X_test en DataFrames si no lo son
            X_val_df = pd.DataFrame(X_val) if not isinstance(X_val, pd.DataFrame) else X_val
            X_test_df = pd.DataFrame(X_test) if not isinstance(X_test, pd.DataFrame) else X_test

            # Ajustar las probabilidades del conjunto de validación
            prob_cal = clf.predict_proba(X_val_df)
            if prob_cal.shape[1] == 1:  # Si solo hay una columna (clase positiva)
                prob_cal = np.hstack((1 - prob_cal, prob_cal))  # Añadir la probabilidad para la clase negativa

            va_clf.fit(prob_cal, y_val)

            # Ajustar las probabilidades del conjunto de prueba
            prob_test = clf.predict_proba(X_test_df)
            if prob_test.shape[1] == 1:  # Si solo hay una columna (clase positiva)
                prob_test = np.hstack((1 - prob_test, prob_test))  # Añadir la probabilidad para la clase negativa

            # Generar probabilidades calibradas usando Venn-Abers
            try:
                prob_va = va_clf.predict_proba(prob_test)[:, 1]
            except Exception as e:
                raise ValueError(f"Error en Venn-Abers: {e}")
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
    save_combined_calibration_plot(dataset_id, all_probabilities, np.array(y_true_all))

    # Guardar métricas en un archivo JSON
    metrics_file = os.path.join(output_dir, f"metrics_dataset_{dataset_id}_repeated.json")
    with open(metrics_file, "w") as f:
        json.dump(final_metrics, f, indent=4)
