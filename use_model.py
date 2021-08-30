import cv2
import tensorflow as tf

picture = "/home/elina_uni/Pictures/random_pictures/"

CATEGORIES = ["Fallen", "Fine"]

def prepare(picpath):
    IMG_SIZE = 150  # 50 in txt-based
    img_array = cv2.imread(picpath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("64x3-CNN.model")

prediction = model.predict([prepare(picture + "room.jpg")])

print(CATEGORIES[int(prediction[0][0])])
