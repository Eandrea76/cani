import os
from PIL import Image
import numpy as np
from sklearn.decomposition import PCA
import glob


def load_and_resize_image(image_path, size=(32, 32)):
    img = Image.open(image_path).convert('RGB') # L for grayscale
    img = img.resize(size)
    return img

'''
Library for dog classification.
'''

def resize_dataset_dog(dataset_path, size = (32, 32)):

    for im_path in glob.glob(os.path.join(dataset_path, "*/*")):
        im = load_and_resize_image(im_path, size)
        label = im_path.split("/")[1]
        os.makedirs(os.path.join(dataset_path + "_resized", label), exist_ok=True)
        path_dir = im_path.replace(dataset_path, dataset_path + "_resized")
        im.save(path_dir)


def create_dataset_dog(dataset_path):
    dataset, labels = [], []
    for im_path in glob.glob(os.path.join(dataset_path, "*/*")):
        img = Image.open(im_path).convert('RGB')

        dataset.append(np.array(img).flatten())
        labels.append(im_path.split("/")[1])
    
    return dataset, labels


# resize_datase_dog("dog_emotion")
dataset, labels = create_dataset_dog("dog_emotion_resized")
print(type(dataset[0]))
print(dataset[0].shape)
pca = PCA(n_components=100)
new = pca.fit_transform(dataset)
print(new.shape)






