import pandas as ps
import numpy as np
from PIL import Image    # caricato perchè non mi attivava la funzione Image di Ale
from library import resize_dataset_dog    # caricare il datasete e trasformare

data_set_path="emozione_cane/angry"
data_cani_piccolo= resize_dataset_dog(data_set_path)
print(data_cani_piccolo)

