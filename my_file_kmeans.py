import utils
from Kmeans import KMeans
import matplotlib.pyplot as plt

def test_single_image_colors(image_path, k=None):
    # 1. Carreguem la imatge des de la carpeta 'images'
    # 'utils.get_imgs' acostuma a retornar una llista d'imatges
    img = utils.get_imgs(image_path)[0]
    
    # 2. Inicialitzem el nostre objecte KMeans
    # Si k és None, buscarem la millor K automàticament
    kmeans = KMeans(img)
    
    if k is None:
        print("Buscant la millor K per a la imatge...")
        kmeans.find_bestK(max_K=10)
        print(f"La millor K trobada és: {kmeans.K}")
    else:
        kmeans.K = k
    
    # 3. Executem l'algorisme
    kmeans.fit()
    
    # 4. Obtenim els colors en format text
    colors = utils.get_color_prob(kmeans.centroids)
    noms_colors = [utils.colors[i] for i in colors.argmax(axis=1)]
    
    # 5. Resultats
    print(f"Colors detectats: {noms_colors}")
    print(f"WCD Final: {kmeans.withinClassDistance()}")
    
    # Visualització (opcional)
    plt.imshow(img)
    plt.title(f"Colors: {', '.join(noms_colors)}")
    plt.show()

if __name__ == "__main__":
     test_single_image_colors('images/0.jpg', k=3)