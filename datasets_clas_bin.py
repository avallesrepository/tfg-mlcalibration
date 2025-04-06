import openml

# Lista de datasets proporcionada
dataset_ids = [3, 15, 29, 31, 37, 38, 44, 316, 953, 958, 962, 978, 1067, 1462, 1462, 1464, 1487, 1494, 1510, 4134, 4134, 40701]
# Verificación de los datasets
binary_classification_datasets = []
non_binary_datasets = []

for dataset_id in dataset_ids:
    try:
        dataset = openml.datasets.get_dataset(dataset_id)
        _, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
        # Si solo hay dos clases en el objetivo, es binario
        if len(set(y)) == 2:
            binary_classification_datasets.append(dataset_id)
        else:
            non_binary_datasets.append(dataset_id)
    except Exception as e:
        print(f"Error con el dataset {dataset_id}: {e}")

print("Binary Classification Datasets:", binary_classification_datasets)
print("Non-Binary Datasets:", non_binary_datasets)
