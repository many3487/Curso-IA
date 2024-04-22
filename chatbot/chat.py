import random
import json
import pickle
#sirve para trabajar con archivos que podamos guardar
import numpy as np

import nltk
#sirve para trabajaar con palabras 
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
#crear modelo de la red neuronal
from keras.optimizers import SGD
#optimizador

lemmatizer = WordNetLemmatizer()
intents =json.loads(open("intents.json").read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
#archivos para que funcione 

words =[]
classes=[]
documents =[]
ignore_letters=["?","!","¿",".",",","#"]#lista con las letras que no interesan o que se quiere que se ignore


#se crean las listas
for intent in intents['intents']:
    for pattern in intent['patterns']:#lo que harpa es que entrara a cada uno de los intentos y luego a cada uno de los patrones
        word_list = nltk.word_tokenize(pattern)#lo que hace es que se comvierte mejor de 1 a 0
        words.extend(word_list)#a la lista de palabras se va a añador todas las que han pasado por esta función
        documents.append((word_list, intent["tag"]))#en esta lista se relacione las palabras que se añadieron al identificador que pertenecen
        if intent["tag"] not in classes: # si no se encuentra el identificador en la lista de clases que simplemente lo añada
            classes.append(intent["tag"])

words =[lemmatizer.lemmatize(word)for word in words if word not in ignore_letters]# va a defirnir la lista en una serie de lista comn palabras simplificadas también que los añada o que no los añada siu contiene algún simbolo que se ha pedido antes
 #ordenar las palabras y convertirlas a un set
words = sorted(set(words))


#se guardas las palabras en un archivo 
pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))
#se crea la lista para hacer el entrenamiento
training=[]
#se crea la salida
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #para cada palabra dentro de esta lista añadir un 1 y la palabra que esta en word patter
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Crear una nueva lista de ceros para cada iteración
    output_row = [0] * len(classes)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])
output_empty=[]*len(classes)
# for document in documents:
#     bag =[]
#     word_patterns=document[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)
#         #se hace un bucle por todos los patrones que se tienen que se reconozcan y se hace una lista en el caso que 1 si las palabras pertenecen al patron y   si no pertenecesn
#     output_row = list(output_empty)
#     output_empty = [0] * len(classes)
#     # esta tomando el input en todas las clases  que se tieen con 1 y el primer indice del documento yu lo comvierte en uno
#     training.append([bag, output_row])
#     #se crea la lista de las palabras y el outputrow

random.shuffle(training)
#se convierte en array
training = np.array(training)
print(training)



