import pickle
import numpy as np
from utils_data import read_dataset
from KNN import KNN
import matplotlib.pyplot as plt


def reducte_size(images):
    return images[:, ::2, ::2]


def Get_class_weighted(knn, k):
    veins_retallats = knn.neighbors[:, :k]
    distancies_retallades = knn.distancies[:, :k]

    preds = []
    for i in range(len(veins_retallats)):
        veins_actuals = veins_retallats[i]
        distancies_actuals = distancies_retallades[i]


        pesos = 1.0/(distancies_actuals + 1e-5)
        classes_unique = np.unique(veins_actuals)
        millor_classe = None
        maxim_pes = -1
        for c in classes_unique:
            pes_total_classe = np.sum(pesos[veins_actuals == c])
            if pes_total_classe > maxim_pes:
                maxim_pes = pes_total_classe
                millor_classe = c
        preds.append(millor_classe)

    return np.array(preds)


def Get_shape_accuracy_weigted(knn, test_labels, k):
    predictions = Get_class_weighted(knn, k)
    correct_predictions = np.sum(predictions == test_labels)
    accuracy = correct_predictions / len(test_labels)
    return accuracy, predictions


def Get_shape_accuracy(knn, test_labels, k):

    veins_retallats = knn.neighbors[:, :k]

    P_test = veins_retallats.shape[0]
    predictions = np.empty(P_test, dtype=knn.labels.dtype)

    for i in range(P_test):
        veins_actuals = veins_retallats[i]  # Fem servir els veïns ja retallats
        valors, comptes = np.unique(veins_actuals, return_counts=True)
        max_vots = np.max(comptes)
        guanyadors = valors[comptes == max_vots]
        if len(guanyadors) > 1:
            for v in veins_actuals:
                if v in guanyadors:
                    predictions[i] = v
                    break
        else:
            predictions[i] = guanyadors[0]

    correct_predictions = np.sum(predictions == test_labels)
    accuracy = correct_predictions / len(test_labels)
    return accuracy, predictions


def Get_shape_accuracy_diversos(knn, test_imgs, test_labels, minim, maxim):
    list_return = []
    for k in range(minim, maxim):
        knn.get_k_neighbours(test_imgs, k)
        preds = knn.get_class()
        number_of_accurates = np.sum(preds == test_labels)
        accuracy = number_of_accurates / len(test_labels)
        packet = (k, accuracy)
        list_return.append(packet)
    return list_return


def Retrieval_by_shape(knn, train_imgs, test_imgs, query_string, k, min_percentatge):


    knn.get_k_neighbours(test_imgs, k)
    predictions = knn.get_class()

    unique_classes = np.unique(predictions)
    print("Classes predicted in this batch:", unique_classes)
    print("Searching for class:", query_string)
    retrieved_results = []
    for i in range(len(predictions)):
        if predictions[i] == query_string:
            veins_actuals = knn.neighbors[i]
            vots_coincidents = np.sum(veins_actuals == query_string)
            percentatge = vots_coincidents / k
            if percentatge >= min_percentatge:
                indices_of_predictions = knn.neighbour_index[i]
                neighbour_images = train_imgs[indices_of_predictions]
                image_pack = (
                    test_imgs[i], predictions[i], neighbour_images, percentatge, knn.neighbors[i])
                retrieved_results.append(image_pack)


    retrieved_results.sort(key=lambda x: x[3], reverse=True)
    return retrieved_results


if __name__ == "__main__":
    from utils_data import read_dataset
    from KNN import KNN
    import matplotlib.pyplot as plt
    import numpy as np

    print("--- CARREGANT DADES ---")
    train_imgs, train_labels, _, test_imgs, test_labels, _ = read_dataset(
        root_folder='../images/', 
        gt_json='../test/gt.json'
    )

    train_imgs_reduced = reducte_size(train_imgs)
    test_imgs_reduced = reducte_size(test_imgs)

    knn_orig = KNN(train_imgs, train_labels)
    knn_red = KNN(train_imgs_reduced, train_labels)

    print("--- CALCULANT MATRIUS DE DISTÀNCIES (Pre-càlcul) ---")
    k_maxim = 49
    knn_orig.get_k_neighbours(test_imgs, k_maxim)
    knn_red.get_k_neighbours(test_imgs_reduced, k_maxim)

    print("--- EXECUTANT EXPERIMENTS ---")
    k_range = list(range(1, 50))
    accuracies_normal = []
    accuracies_weighted = []
    accuracies_reduced_normal = []
    accuracies_reduced_weighted = []

    for k in k_range:
        acc_n, _ = Get_shape_accuracy(knn_orig, test_labels, k)
        acc_w, _ = Get_shape_accuracy_weigted(knn_orig, test_labels, k)
        accuracies_normal.append(acc_n)
        accuracies_weighted.append(acc_w)

        acc_n_r, _ = Get_shape_accuracy(knn_red, test_labels, k)
        acc_w_r, _ = Get_shape_accuracy_weigted(knn_red, test_labels, k)
        accuracies_reduced_normal.append(acc_n_r)
        accuracies_reduced_weighted.append(acc_w_r)

    print("--- GENERANT GRÀFICA ---")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies_normal, label='Original (Majoria simple)')
    plt.plot(k_range, accuracies_weighted, label='Original (Ponderat)', linestyle='--')
    plt.plot(k_range, accuracies_reduced_normal, label='Reduït (Majoria simple)')
    plt.plot(k_range, accuracies_reduced_weighted, label='Reduït (Ponderat)', linestyle='--')
    
    plt.title('Comparativa Precisió KNN (Original vs Reduït)')
    plt.xlabel('K (Veïns)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
