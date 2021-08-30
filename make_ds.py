import numpy as np
import os
import cv2
from tqdm import tqdm
import random
import pickle

FOLDER = "/home/elina_uni/Documents/snh_things/dataset/pictures"
CATEGORIES = ["Fallen", "Fine"]

for category in  CATEGORIES:
    path = os.path.join(FOLDER, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)

IMG_SIZE_W = 150

training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(FOLDER,category)
        class_num = CATEGORIES.index(category)

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE_W, IMG_SIZE_W))
                training_data.append([new_array, class_num])
            
            except Exception as e:
                pass

create_training_data()

print(len(training_data))

random.shuffle(training_data)

'''
for sample in training_data[:10]:
    print(sample[1])
'''

X = []
Y = []

for features, label, in training_data:
    X.append(features)
    Y.append(label)

print(X[0].reshape(-1, IMG_SIZE_W, IMG_SIZE_W, 1))
print(Y)

X = np.array(X).reshape(-1, IMG_SIZE_W, IMG_SIZE_W, 1)
Y = np.array(Y)

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("Y.pickle","wb")
pickle.dump(Y, pickle_out)
pickle_out.close()
