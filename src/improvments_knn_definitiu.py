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


        weights = 1.0/(distancies_actuals + 1e-5)
        classes_unique = np.unique(veins_actuals)
        millor_classe = None
        maxim_pes = -1
        for c in classes_unique:
            pes_total_classe = np.sum(weights[veins_actuals == c])
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

import numpy as np

def normalitza_z_score_grayscale(images):
    #assegurem que les imatges passades com a parametre estan representades en float64
    #a contianucio calculem la mitjana de brillantor de cadascuna de les imatges i es guarda dins de la llista mitjana(array numpy)
    #despres es calcula la desviacio de cada imatge, es a dir, com es difereixen la brillantor dels diferents pixels 
    #Es a dir, desviacio mesura el contrasnt de la imatge, ens diu com de lluny els pixels de la imatge estan de la propia mitjana de brillantor.
    #si tenim una desviacio baixa: els pixels tenen gariabe la mateixa brillantor, es a dir, no hi ha molta diferncia en brillantor, tots son molt grisos per exemple, o molt blancs.
    #si tenim una desviacioi alta hi ha molt de contrast, hi han molts pixels amb diferents nivells de brillantor: un blancs i altres grisos
    #Al final estem forçant el knn a fixar-se nomoés en la forma geometrica i no tant en les diferentes tonalitats de l'escala de grisos que tenen les imatges.
    """
    Primer restem de cada pixel de la imatge la mitjana de brilantor, estem centrant el brillantor a zero, quan abans estava centrat en la mitjana de brillantor. Si per exemple la imatge s'ha fet amb molta llum, aquest calcul ho elimina: estem noramlitzant totes les imatges perque estiguin centrades a brillantor zero.
    Despres es divideix entre la desviació, s'iguala el contrast
    """
    imgs_float = images.astype(np.float64)
    mitjana = np.mean(imgs_float, axis=(1, 2), keepdims=True)
    desviacio = np.std(imgs_float, axis=(1, 2), keepdims=True)
    images_normalitzades = (imgs_float - mitjana) / (desviacio + 1e-5)
    return images_normalitzades

if __name__ == "__main__":
    from utils_data import read_dataset
    from KNN import KNN
    import matplotlib.pyplot as plt
    import numpy as np
    from utils_data import crop_images
    print("--- CARREGANT DADES ---")
    train_imgs, train_labels, _, test_imgs, test_labels, _ = read_dataset(root_folder='../images/', gt_json='../test/gt.json')

    k_maxim = 99


    upper_train = [(5,5)]*len(train_imgs)
    lower_train = [(55,75)]*len(train_imgs)
    upper_test = [(5,5)]*len(test_imgs)
    lower_test = [(55,75)]*len(test_imgs)
    croped_train_imgs=crop_images(train_imgs, upper_train, lower_train)
    croped_test_imgs= crop_images(test_imgs, upper_test, lower_test)
    croped_train_reduced_imgs = reducte_size(croped_train_imgs)
    croped_test_reduced_imgs = reducte_size(croped_test_imgs)

    train_imgs_gray = normalitza_z_score_grayscale(train_imgs)
    test_imgs_gray = normalitza_z_score_grayscale(test_imgs)

    knn_gray = KNN(train_imgs_gray, train_labels)
    knn_gray.get_k_neighbours(test_imgs_gray, k_maxim)

    train_imgs_reduced = reducte_size(train_imgs)
    test_imgs_reduced = reducte_size(test_imgs)

    knn_orig = KNN(train_imgs, train_labels)
    knn_red = KNN(train_imgs_reduced, train_labels)
    knn_croped = KNN(croped_train_imgs, train_labels)
    knn_croped_reduced = KNN(croped_train_reduced_imgs, train_labels)


    print("--- CALCULANT MATRIUS DE DISTÀNCIES (Pre-càlcul) ---")
    knn_orig.get_k_neighbours(test_imgs, k_maxim)
    knn_red.get_k_neighbours(test_imgs_reduced, k_maxim)
    knn_croped.get_k_neighbours(croped_test_imgs,k_maxim)
    knn_croped_reduced.get_k_neighbours(croped_test_reduced_imgs, k_maxim)
    print("--- EXECUTANT EXPERIMENTS ---")
    k_range = list(range(1, 100))
    accuracies_normal = []
    accuracies_weighted = []
    accuracies_reduced_normal = []
    accuracies_reduced_weighted = []
    acc_croped_normal, acc_croped_weighted = [],[]
    acc_croped_red_normal, acc_croped_red_weighted = [],[]
    accuracies_gray_normal = []
    accuracies_gray_weighted = []

    for k in k_range:
        acc_g_n, _ = Get_shape_accuracy(knn_gray, test_labels, k)
        acc_g_w, _ = Get_shape_accuracy_weigted(knn_gray, test_labels, k)
        accuracies_gray_normal.append(acc_g_n)
        accuracies_gray_weighted.append(acc_g_w)



        acc_n, _ = Get_shape_accuracy(knn_orig, test_labels, k)
        acc_w, _ = Get_shape_accuracy_weigted(knn_orig, test_labels, k)
        accuracies_normal.append(acc_n)
        accuracies_weighted.append(acc_w)

        acc_n_r, _ = Get_shape_accuracy(knn_red, test_labels, k)
        acc_w_r, _ = Get_shape_accuracy_weigted(knn_red, test_labels, k)
        accuracies_reduced_normal.append(acc_n_r)
        accuracies_reduced_weighted.append(acc_w_r)
        #a continaucio tenim per cropped:

        ac_cn, _ = Get_shape_accuracy(knn_croped, test_labels, k)
        ac_cr, _ = Get_shape_accuracy(knn_croped_reduced, test_labels, k)
        ac_cnw, _ = Get_shape_accuracy_weigted(knn_croped, test_labels, k)

        ac_crw, _ = Get_shape_accuracy_weigted(knn_croped_reduced,test_labels,k)

        acc_croped_normal.append(ac_cn)
        acc_croped_weighted.append(ac_cnw)
        acc_croped_red_normal.append(ac_cr)
        acc_croped_red_weighted.append(ac_crw)

    print("--- TAULA AMB MODIFICACIO DE PARAMETRES I FUNCIONS DE MILLORAMENT---")
    print("-" * 145)
    print(f"{'K':<5} | {'Normal':<12} | {'Pesat':<12} | {'Reduit':<12} | {'Red. Pesat':<12} | {'Cropped':<12} | {'Crop. Pesat':<12} | {'Crop.Red.':<12} | {'Crop.Red. P.':<12} | {'Gray Normal':<12} | {'Gray Pesat':<12}")
    print("-" * 145)
    for k, norm, weight, r_norm, r_weight, c_norm, c_weight, cr_norm, cr_weight, gray, gray_weight in zip(k_range, accuracies_normal, accuracies_weighted, accuracies_reduced_normal, accuracies_reduced_weighted, acc_croped_normal, acc_croped_weighted, acc_croped_red_normal, acc_croped_red_weighted, accuracies_gray_normal, accuracies_gray_weighted):
        print(f"{k:<5} | {norm:<12.4f} | {weight:<12.4f} | {r_norm:<12.4f} | {r_weight:<12.4f} | {c_norm:<12.4f} | {c_weight:<12.4f} | {cr_norm:<12.4f} | {cr_weight:<12.4f} | {gray:<12.4f} | {gray_weight:<12.4f}")
    print("-" * 145)
    print("--- RESUM FINAL ---")
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies_normal, label="Orig Normal")
    plt.plot(k_range, accuracies_weighted, label="Orig Weighted", linestyle='--')
    plt.plot(k_range, accuracies_reduced_normal, label="Reduït Normal")
    plt.plot(k_range, accuracies_reduced_weighted, label="Reduït Weighted", linestyle='--')

    plt.plot(k_range, acc_croped_normal, label="Croped")
    plt.plot(k_range, acc_croped_weighted, label="Croped Weighted", linestyle='--')
    plt.plot(k_range, acc_croped_red_normal, label="Croped reduced")
    plt.plot(k_range, acc_croped_red_weighted, label="Croped reduced Weighted", linestyle='--')
    plt.plot(k_range, accuracies_gray_normal, label="Grayscale")
    plt.plot(k_range, accuracies_gray_weighted, label="Grayscale Weighted", linestyle='--')

    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Comparativa KNN: Original vs Reduït vs Croped')
    plt.grid(True)
    plt.show()
