import pickle 
import numpy as np
from KNN import KNN


def reducte_size(images):
	#aquesta funcio nomes redueix ell el numero de pixels que processem, millorant llavors l'eficiencia de l'algorisme pero perdent detall de les imatges.
	return images[:,::2,::2]

def Get_class_weighted(knn):


"""
Aquesta funcio serveix per poder corregir un error tipic de knn: votació per majoria erronia.
Si tenim que una imatge te una distancia euclidiana mes propera, li afegim mes pes que dos imatges que no estan molt aprop pero que son de la mateixa classe(abans soritira error, perque es feia per vot de majoria, pero aixo s'elimina amb ponderació)
"""
#cal tenir en compte que ja es fa tot suposant que s'han carregat els veins i els seus labels.

	preds = []
	for i in range(len(knn.neighbors)):
		#the first for loop iterates untill we have gone threw each test_img and its neighbours
		veins_actuals = knn.neighbors[i]
		#cal enrecordar que neighbours es una matriu de dos dimensions, on les files represten diferentes imatges i les columnes son els diferents labels dels k veins que hem trobat
		#distancies_actuals guarda les distancies euclidinaes de tots els k veins de totes les imates de test_imgs
		distancies_actuals = knn.distancies[i]
		#calculem el seu pes ponderat: 1/distancia euclidiana for each preediction of the imaeg that we are looking at right now
		#pesos is going to be an array that contains the "pes ponderat" for each label, taking into account the euclidean distance between the current neighbour and the test_img that we are on right now
		pesos = 1.0/(distancies_actuals + 1e-5)
		#we go through the hole matrice and now we have a one dimension array that contains all of the labels that have appeared in all predictions of all test_imgs
		classes_unique = np.unique(veins_actuals)
		millor_classe = None
		maxim_pes = -1
		#we go through each class
		for c in classes_unique:
			#sumem els pesos ponderats de tots els elements on veins_actuals (labels) son equivalents al label que estem mirant en el for loop, si es superior al maxim que hem vist fins ara s'actauliza
			pes_total_classe = np.sum(pesos[veins_actuals==c])
			if pes_total_classe > maxim_pes:
				maxim_pes = pes_total_classe
				millor_classe=c
		preds.append(millor_classe)

	#al final es retorna un array de dimensio len(test_imgs) que conté el label ponderat
	return np.array(preds)

def Get_shape_accuracy_weigted(knn, test_imgs, test_labels, k):

	knn.get_k_neighbours(test_imgs, k)

	preds = Get_class_weighted(knn)

	number_of_accurates = np.sum(preds == test_labels)

	accuracy = number_of_accurates/len(test_labels)

	return (accuracy, preds)


def Get_shape_accuracy(knn, test_imgs, test_labels, k):
	#aquesta funcio ens serveix per determinar si les prediccions son certes o falses
	#primer calculem les prediccions que genera l'algorisme KNN i les guardem: una matriu de 2 dimensions, on cada fila té k columnes i cada fila és una imatge de test_imgs
	knn.get_k_neighbours(test_imgs, k)
	preds = knn.get_class()
	#despres d'obtenir les prediccions, mirem si el que s'ha calculat és equivalent a test_labels(les classes/etiquetes verdaderes)

	#enrecorda que dins de preds és un array de numpy 1 dimensió on cada index correspont a una imatge de test_imgs
	#en la instrucció seguent es realitza una comparació element per element entre el que s'ha predit i les classes reals de les imatges de test.
	number_of_accurates = np.sum(preds == test_labels)
	#dins de number_of_accurates tenim ja la suma dels True
	#al final ho dividim tot entre el numero d'elements dins de test_labels
	accuracy = number_of_accurates/len(test_labels)
	return (accuracy, preds)


def Get_shape_accuracy_diversos(knn, test_imgs, test_labels, minim, maxim):
	#aquetsa funcio, com ja comentat en el main, es similar a Get_shape_accuracy, pero en comptes de passar una k, es realiza un for loop per veure com va variant la precisió de l'algorsime a mesura que va incrementant el rang de veins que nosatlres acceptem
	list_return  = []
	for k in range(minim, maxim):
		knn.get_k_neighbours(test_imgs, k)
		preds = knn.get_class()
		number_of_accurates = np.sum(preds == test_labels)
		accuracy = number_of_accurates / len(test_labels)
		packet = (k, accuracy)
		list_return.append(packet)
	return list_return

def Retrieval_by_shape(knn, train_imgs, test_imgs, query_string, k, min_percentatge):

	#primermaent carregeum els veins i les prediccions de l'algorisme
	#despres a continaucio es crea un array de np amb les prediccions uniques: nomes un de cada
	knn.get_k_neighbours(test_imgs, k)
	predictions = knn.get_class()
	unique_classes = np.unique(predictions)
	print("Classes predicted in this batch:", unique_classes) 
	print("Searching for class:", query_string)
	retrieved_results = []
	#aquesta funcio es basa en trobar quina de les prediccions equival al query_string i obtenir les imatges que superen un minim de percentatge segons els veins de la imatge test.
	#passem per totes les prediccions
	#si la prediccio en curs es igual a query_string, entrem dins del if, hem trobat un cas equivalent
	for i in range(len(predictions)):
		if predictions[i] == query_string:
			#despres de trobar una prediccio igual a query_string, obtenim tots els altres veins que l'algorisme ha trobat
			#es mira quins son els vots coincidents/vots que son igual al query_string
			#i finalment calculem el percentatge de vots que son equivalents a query_string
			veins_actuals = knn.neighbors[i]
			vots_coincidents = np.sum(veins_actuals == query_string)
			percentatge = vots_coincidents / k
			#es mira si el percentatge supera el minim, si es que si es prepara la tuple amb el contingut seguent:
			#tuple:
			#la imatge de test corresponent, l'etiqueta que s'ha preduit, totse les imatges veines a test_imgs[i], el percentatge i els labels de les imatges veines de test_imgs[i]
			if percentatge >= min_percentatge:
				indices_of_predictions = knn.neighbour_index[i]
				neighbour_images = train_imgs[indices_of_predictions]
				image_pack = (test_imgs[i], predictions[i], neighbour_images, percentatge,knn.neighbors[i])
				retrieved_results.append(image_pack)


	#al final ho retornem tot organitzat tenint en compte el tercer element: percentatge
	#cal enrecordar que aquesta funcio nomes retorna els casos on la predicció de l'algorsime resulta en query_string i si es supera el treshold the min_percentatge (resta de veins)
	#tambe cal tenir en compte que si la predicció obtinguda es erronia pero igualment equivalent a query_string, obtindrem resutlats erronis. No es que tenim un problema en la impelmentació
	#... d'aquesta funció, sinó en el propi algorisme de KNN: s'arrosegua un possible error en l'algorisme KNN. Tot i aixo hem definit una funció que permetrà corregir o al meny intentar corregir aquest possible error del knn: Get_class_weighted

	retrieved_results.sort(key=lambda x: x[3], reverse=True)
	return retrieved_results

if __name__ == "__main__":

	#a continuacio definim el main basic pel knn.
	#primerament a partir del fitxer test_cases_knn.pkl, carreguem tot el contignut de les iamtges/etiquetes
	#obtenim les imatges supervisades/les que ja estan etiquetades
	#obtenim les imatges de test per comprovar l'algorisme
	#a part de les dades/pixels de les imatges, també obtenim les seves etiquetes corresponents
	with open('../test/test_cases_knn.pkl', 'rb') as f:
		test_cases = pickle.load(f)
	train_imgs = test_cases['input'][0][0]
	train_labels = test_cases['input'][0][1]
	test_imgs = test_cases['test_input'][0][0]
	test_labels  = test_cases['get_class'][0]

	#a continuacio es crea l'objecte knn:
	knn = KNN(train_imgs, train_labels)

	#a continuacio presentem les imatges de test:

	for class_image in test_labels:
		print("image label:",class_image)

	#a continuacio tenim el codi per veure el resultat de les funcions que nosaltres hem implementat

	#Get_shape_accuracy

	print("TESTING Get_shape_accuracy")

	k = 3

	accuracy_defined_k, predictions = Get_shape_accuracy(knn, test_imgs, test_labels, k)

	print("We are testing the following images with the corresponding labels")

	print("accuracy:", accuracy_defined_k)

	print("labels: ")

	for real_label, predicted_label in  zip(test_labels, predictions):

		print(f'real label: {real_label} i la seva predicció: {predicted_label}')




	#Get_shape_accuracy_diversos

	print("TESTING Get_shape_accuracy_diversos")

	#hem definit una segona funcio, similar a Get_shape_accuracy, pero en comptes de passar sempre el numero de veins que volem per cada imatge de test, dins de la funcio tenim un for loop que va recurrent
	#... entre 1 i 50: aquesta funcio ens permet visualitzar com va variant la precisió dins del for loop a mesura que el rang de veins que acceptem va incrementant

	#la funcio retorna una llista de tuples, on en cada tuple tenim el valor k i la precisió calculada

	minim = 1

	maxim = 50

	list_accuracy_diversos = Get_shape_accuracy_diversos(knn, test_imgs, test_labels, minim, maxim)

	for element_k, element_p in list_accuracy_diversos:

		print(f'valor k: {element_k}, precisió: {element_p}')



	#Retreival_by_shape:

	print("TESTING Retreival_by_shape")

	#definicio dels paramtres per Retieval_by_shape:

	query_string = test_labels[2] #el valor que escollim pot ser qualsevol

	percentatge_minimum = 0.0

	k = 3

	image_packet = Retrieval_by_shape(knn, train_imgs, test_imgs, query_string, k, percentatge_minimum)

	for raw_test, label_prediction,raw_neighbour, precision, label_neighbours in image_packet:
		print(f"  - Imatge de test predita com a '{label_prediction}' amb una coincidència del {precision*100:.1f}%")
	        print(f"    Els seus veïns han estat etiquetats com: {label_neighbours}")


	#Get_class_weighted

	print("TESTING Get_class_weighted")

	#primer cal prepara els neighbours i les distancies euclidianes per poder cirdar Get_class_weighted

	k = 3

	knn.get_k_neighbours(test_imgs,k)

	#no cal cridar .get_class, ja hem carregat els veins
	array_prediccions_weighted = Get_class_weighted(knn)

	#array_prediccions es un array de una dimensio, on cada index correspont per cada imatge

	for element_pes, temporal_label in zip(array_prediccions_weighted,test_labels):

		print(f'weighted prediction: {element_pes}, real class: {temporal_label}')


	#ara caldra fer un analasi de comparacio entre weighted i sense pes euclidia
	#per poder fer aixo, cal primer calcular la precisió del weighted
	for k in range(50):

		normal_accuracy, normal_prediction = Get_shape_accuracy(knn, test_imgs, test_labels, k)

		weighted_accuracy, weighted_prediction = Get_shape_accuracy_weigted(knn, test_imgs, test_labels, k)

		i = 0

		for normal, weighted in zip(normal_accuracy, weighted_accuracy):

			print("accuracy without any weight: ", normal)
			print("prediction without any weight:", normal_prediction[i])
			print("accuracy with weight: ", weighted)
			print("prediction with weight: ", weighted_prediction[i])
			print("the actual real label: ", test_labels[i])
