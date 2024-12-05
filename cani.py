import pandas as pd
import numpy as np
from PIL import Image    # caricato perchè non mi attivava la funzione Image di Ale
from libreria_immagine import resize_dataset_dog,create_dataset_dog    # caricare il datasete e trasformare
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

##data_set_path="emozione_cane"
data_set_path="images"
#resize_dataset_dog(data_set_path,size=(32,32))

# provo un metodo pre risparmiare memoria
# per converitire i numere in float32

# data_set = [image.flatten() for image in data_set_path]
# data_set = np.array(data_set, dtype=np.float32)


data_set, labels=create_dataset_dog(data_set_path)
# # # print(data_set[])
# # # print(labels[])
# # pca = PCA(n_components=100) 
# # data_set = pca.fit_transform(data_set)


# appiattimento della funzione
data_set = np.array([image.flatten() for image in data_set], dtype=np.float32)


#normaliziamo i dati
scaler = StandardScaler()
data_set = scaler.fit_transform(data_set)

# usiamo la pca con un numero piu basso
pca = PCA(n_components=50)  
data_set = pca.fit_transform(data_set)






# # dataset_norm = [(image - image.min()) / (image.max() - image.min()) for image in data_set]
# # print(dataset_norm[0])

# Codifica delle etichette
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)



#codificazione delle etichette
# label_encoder = LabelEncoder()
# label_encoder.fit(labels)
##labels = label_encoder.transform(labels)
#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data_set, labels, random_state=30, test_size = 0.2)


# prova con svc
# svm = SVC(kernel='rbf')
# svm.fit(X_train, y_train)

# y_pred = svm.predict(X_test)


# prova con clf
# clf = RandomForestClassifier(max_depth=15, random_state=0)
# clf.fit(X_train,y_train)
# y_pred=clf.predict(X_test)


# prova con KNeighbor
neigh = KNeighborsClassifier(n_neighbors=7)
neigh.fit(X_train, y_train)

y_pred=neigh.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuratezza: {accuracy:.2f}')
