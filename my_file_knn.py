import utils_data
from KNN import KNN
import matplotlib.pyplot as plt
import numpy as np

def test_single_prediction(image_index=0):
    # 1. Carreguem totes les dades d'entrenament i test
    # utils_data s'encarrega de llegir els fitxers de la carpeta 'images'
    train_imgs, train_labels, test_imgs, test_labels = utils_data.read_dataset()
    
    # 2. Inicialitzem el KNN amb les dades d'entrenament
    knn = KNN(train_imgs, train_labels)
    
    # 3. Triem una imatge de test específica per provar
    imatge_a_provar = test_imgs[image_index]
    etiqueta_real = test_labels[image_index]
    
    # 4. Fem la predicció
    # Recorda que predict espera una matriu (per això fem [None, ...])
    prediccio = knn.predict(imatge_a_provar[np.newaxis, ...], k=5)
    
    # 5. Resultats
    print(f"Etiqueta Real: {etiqueta_real}")
    print(f"Predicció del KNN: {prediccio[0]}")
    
    # Visualització
    plt.imshow(imatge_a_provar, cmap='gray')
    plt.title(f"Real: {etiqueta_real} | Pred: {prediccio[0]}")
    plt.show()

if __name__ == "__main__":
    # Provem amb la primera imatge del conjunt de test
    test_single_prediction(image_index=0)