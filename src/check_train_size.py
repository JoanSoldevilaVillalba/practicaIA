import pickle
import numpy as np
from utils_data import read_dataset

# Carrega el dataset complet per referència
train_imgs, _, _, _, _, _ = read_dataset(root_folder='./images/', gt_json='./images/gt.json')
print(f"Nombre d'imatges d'entrenament del dataset complet: {len(train_imgs)}")

# Carrega i inspecciona test_cases_knn.pkl
with open('./test/test_cases_knn.pkl', 'rb') as f:
    test_cases = pickle.load(f)

print("Claus del diccionari test_cases:", list(test_cases.keys()))
for ix, (train_imgs_case, train_labels) in enumerate(test_cases['input']):
    print(f"Cas {ix}: Nombre d'imatges d'entrenament = {train_imgs_case.shape[0]}")
    # Comprova que coincideix amb el dataset complet
    if train_imgs_case.shape[0] == len(train_imgs):
        print(f"  -> Coincideix amb el dataset complet.")
    else:
        print(f"  -> No coincideix (possiblement dades sintètiques o subset).")