import pickle
import numpy as np
from KNN import KNN

def Get_shape_accuracy(knn):
	#aquesta funcio serveix per saber si les prediccions que ens retorna l'algorisme implementat s'acosten a la realitat, es a dir al "Ground-Truth"
	for k in range(1,50): 
	    knn.get_k_neighbours(test_imgs, k)
	    preds = knn.get_class()
	    number_of_accurates = np.sum(preds == test_labels)
	    accuracy = number_of_accurates / len(test_labels)
	    print("number of accurate predictions: ", number_of_accurates)
	    print(f"K: {k}, Accuracy: {accuracy}")


if __name__=="__main__":

	#primer es carrega i es processa les imatges per l'algorisme supervisat: knn
	with open('./test/test_cases_knn.pkl', 'rb') as f:
		test_cases  = pickle.load(f)
	#a continuacio obtenim les dades de les imatges que ja tenen etiquetes, es a dir, les imatges ja preparades. Ens basarem amb aquestes imatges per calcular els veins mes propers a les imatges de test.
	train_imgs  = test_cases['input'][0][0] # test cases is a dictionary. What is 'input'? it is a key that corresponds to a list of tuples. Each tuple has two elements: the first element is the train_imgs and the second element is the train_labels.
	train_labels = test_cases['input'][0][1]
	#despres obtenim les imatges de test, obtenim tant les dades de les imatges de test com els seus labels veraders
	test_imgs = test_cases['test_input'][0][0]
	test_labels = test_cases['get_class'][0]

	#a continuacio creem l'objecte de knn, per inicialitzar l'objecte primer es converteixen les dades de les imatges a arrays d'una dimensio.
	knn = KNN(train_imgs, train_labels)

