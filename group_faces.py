print("INICIO")

import pandas as pd
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2gray
import numpy as np
import os

files = os.listdir('img/')
imgs = []
array = []

for file in files:
    img = imread("img/" + str(file))
    img_flattent = img.flatten()

    imgs.append(img)
    array.append(img_flattent)

model_decomposition = PCA(n_components=2)

new_array_decomposition = model_decomposition.fit_transform(np.array(array))

model_group = HDBSCAN(cluster_selection_epsilon=0)
model_group.fit(new_array_decomposition)

for group in range(np.max(model_group.labels_) + 1):
    try:
        os.mkdir("grupo/" + str(group))
    except:
        print

aux_i = 0

for label in model_group.labels_:
    imsave("grupo/" + str(label) + "/" + str(aux_i) + ".jpg", imgs[aux_i])
    aux_i += 1


print("FIN")