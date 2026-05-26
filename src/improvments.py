import pickle
import numpy as np
from KNN import KNN

"""
En el nostre codi, si per exemple tenim que busquem k = 3, i diem que dos veins son la mateixa peça de roba i 1 dels veins es diferent, surtira guanyannt el label que ha sortit mes, tot i aixo, aquesta metodologia/algorisme pot resultar en resutlats incorrctes/erronis.
Per corregir-ho: votació ponderada, el vto de cada vei valgui mes com a mes a prop estigui:
un vei molt aprop tindra una dsitancia euclidiana molt petita, epr tant la seva inversa sera un valor gran: 1/distancia
un vei llunya tindra una distancia euclidinaa bastant gran, de amera que la seva inversa sera petita.
A continuacio tenim la funcio:
"""
def Get_class_weighted(knn):
"""
A continuacio explicarem que es el que fa cada part de la funcio:
primerament recurrem per l'array de 2 dimensions: passem per cada test_imgs i obtenim els seus k veins
despres a continaucio obtenim les distancies euclidiandes respecte la imatge test que estem observant en la itearció concreta del for loop
despres a continaucio es construeix un array que es diu pesos, que te tots els pesos de tots els k veins
despres es genera una classe_unique, que es una llista sense labels repetits.
s'inicialitzen dos variables per poder anar actualitzant la classe amb mes ponderacio i amb pes maxim
Dins del segon for loop iterem per tots els elements dels labels unics que hem definit abans
sumem tots els pesos que son iguals a la classe/etiqueta que estem observant en el moment
mirem si el pes total suera el maxim_pes
si es que si la millor_classe  es actualitzada.
al final retornem un array que conté les millor classes segons el seu valor ponderat de l'algorise definit dins de la funcio

"""

	for i in range(len(knn.neighbors)):
		veins_actuals = knn.neighbors[i]
		distancies_actuals = knn.distancies[i]
		pesos = 1.0/(distancies_actuals+1e-5) # el 1e-5 es per un cas especific: si un valor de distancies_actuals equival a zero, feriem una divisio per zero, que ens donaria error/undefined
		classes_unique = np.unique(viens_actuals)
		millor_classe = None
		maxim_pes = -1
		for c in classes_unique:
			pes_total_classe = np.sum(pesos[veins_actuals == c])
			if pes_total_classe>maxim_pes:
				maxim_pes = pes_total_classe
				millor_classe = c
		preds.append(millor_classe)
	return np.array(preds)

#enrecroda que train_data es un array de 3 dimensions: la primera dimensio es la imatge i les 2 altres es el grid gris de la imatge: 28x28
#per mimllorar el programa podem reduir la mida de totes les imatges, tot i aixo cal tenir en compte que la reducció s'ha d'aplicar sobre les imatges supervisades com les imatges de test.
def reduce_features(images):
# la implementacio de la funcio es molt simple: primerament cal agafar totes les imatges(:,), despres per l'alçada i amplada anem saltant de dos en dos, de manera que es fa una reducció de factor de 4 en la imatge.
# d'aquesta manera al processar les imatges podrem anar mes rapid. Despres mirarem si aixo empitjora molt a la precisio de l'algorisme o no
	return images[:, ::2, ::2]
def Get_shape_accuracy(knn, test_imgs, test_labels, k):
	"""
	Explicacio dels parametres de la funcio Get_shape_accuracy
		knn: objecte que te les imatges supervisades ja preparades
		test_imgs: son les imatges que aplicarem sobre l'algorisme per veure si l'algorisme es precis o no
		test_labels: son les etiquetes/classes correctes de les imatges
	"""
	#primerament cridem la funció get_k_neigubours per calcular els k veins mes propers.
	#a continuacio obtenim la prediccio final que ha calculat l'algorsme, es a dir, el vei mes proper a cadascuna de les imatges dins de test_imgs
	#despres es calcula el percentatge d'encerts.
	knn.get_k_neighbours(test_imgs, k)
	preds = knn.get_class()
	number_of_accurates = np.sum(preds == test_labels)
	accuracy = number_of_accurates / len(test_labels)
	print("number of accurate predictions: ", number_of_accurates)
	print(f"K: {k}, Accuracy: {accuracy}")
	return accuracy
def Get_shape_accuracy_diversos(knn, test_imgs, test_labels):
	#aquesta funcio es equivalent a Get_shape_accuracy pero en comptes de passar un k com a parametre, es realitza un for loop per veure com va evolucionant la precisió de l'algorisme a mesura que anem incrementant el rang de veins
	for k in range(1, 50):
		knn.get_k_neighbours(test_imgs, k)
		preds = knn.get_class()
		number_of_accurates = np.sum(preds == test_labels)
		accuracy = number_of_accurates / len(test_labels)
		print("number of accurate predictions: ", number_of_accurates)
		print(f"K: {k}, Accuracy: {accuracy}")

def Retrieval_by_shape(knn, train_imgs, test_imgs, query_string, k, min_percentage=0.0):
	"""
	A continuacio expliquem tots els parametres de la funcio Retrival_by_shape:
	primerament l'objete Knn, on ja tenim les imatges supervisades ja processades i preparades per ser utitliades
	train_imgs, llista de imatges ja processades, test_imgs les imatges de prova
	query_string conté l'etiqueta/classe que volem detectar
	k, el numero de veins que volem, es a dir, la precissio que volem
	min_percentatge, es el minim de percentatge que nosatlres podem acceptar.
	"""
	knn.get_k_neighbours(test_imgs, k)
	predictions = knn.get_class()
	#fins aqui, tenim ja els veins de tots els test_imgs i les prediccions, es a dir, els labels que hem pogut identificar de cada image de test_imgs utitlizant el nostre algorisme.
	unique_classes = np.unique(predictions)
	print("Classes predicted in this batch:", unique_classes)
	print("Searching for class:", query_string)
	retrieved_results = []
	#passem per totes les prediccions que hem fet, es a dir, passem per totes els test_img
	for i in range(len(predictions)):
	#en el cas que es igual a query_string, entrem dins del if statment
		if predictions[i] == query_string:
	#obtenim els labels dels veins que nosaltres hem trobat
	#sumem aquells veins que hem trobat que equivalen al query_string per obtenir el percentatge
			veins_actuals = knn.neighbors[i]
			vots_coincidents = np.sum(veins_actuals == query_string)
			percentatge = vots_coincidents / k
	#si el percentatge cau dins del minim, el podem afegir dins de la llista final
	#la llista final es una llista de tuples, on cada tuple es una test_img que ha superat el percentatge minim
	#cada tule conte el seguent:
	#primer element: els pixels de la image que estem provant (test_img)
	#segon element: el label que l'algorisme dissenyat ha assignat
	#tercer element: neighbour_images, una llista de les k imatges (pixels)
	#quart: percentatge, es a dir, el percentatge de les k imatges que son iguals al query_string
	#anem a afegir un cinque: els labels dels veins, per veure de quines classes pertayen per veure el percenatge d'una menra mes visual, aixo es nomes obtenir els veisn calculats d'aquest element:  self.neighbors = self.labels[index_ordenats]
	#tot i aixo, cal tenir en compte que potser la prediccio que ha retornat l'algorisme tampoc es correcta tot i que hem entrat dins del if statment
			if percentatge >= min_percentage:
				indices_of_predictions = knn.neighbour_index[i]
				neighbour_images = train_imgs[indices_of_predictions]
				image_pack = (test_imgs[i], predictions[i], neighbour_images, percentatge,knn.neighbors[i])
				retrieved_results.append(image_pack)
	retrieved_results.sort(key=lambda x: x[3], reverse=True)
	return retrieved_results
if __name__ == "__main__":
	with open('../test/test_cases_knn.pkl', 'rb') as f:
		test_cases = pickle.load(f)
	train_imgs = test_cases['input'][0][0]
	train_labels = test_cases['input'][0][1]
	test_imgs = test_cases['test_input'][0][0]
	test_labels = test_cases['get_class'][0]
	knn = KNN(train_imgs, train_labels)
#	Get_shape_accuracy(knn, test_imgs, test_labels)
	print("\nA continuacio ve el procés que realment volem comprovar")
	print("The labels of the test_images: ")
	for element in test_labels:
		print("image label: ", element)

	print("label that we have chosen:", test_labels[0])
	print("Percentatge of acceptance:", 0.0)
	print("number of neighbours",3)
	image_pack_list = Retrieval_by_shape(knn, train_imgs, test_imgs, test_labels[0], 3, 0.0)
	print("Resultat de retrieval_by_shape obtingut correctament. Mida de la llista:", len(image_pack_list))
	for element in image_pack_list:
		test_img, prediction, neighbour_images, percentatge, list_of_labels = element
		print("-" * 50)
		print(f"Classe predita / cercada: {prediction}")
		print(f"Certesa de la votació (percentatge): {percentatge * 100:.2f}%")
		print(f"Mida de la imatge de test (shape): {test_img.shape}")
		print(f"Quantitat de veïns retornats en el pack: {len(neighbour_images)}")
		print(f"Mida de la matriu de veïns: {neighbour_images.shape}")
		print("llista de les classes dels k veins: ")
		print(list_of_labels)
