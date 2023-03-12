import pickle
import cv2
import os
import tensorflow.keras.models as models
import tensorflow.keras.utils as utils
import tensorflow as tf
from subprocess import run
from random import randrange, choice

load_path = 'saves/faces_model_save_2023_03_10_100_epochs'
model = models.load_model(load_path)

with open(f'{load_path}/class_names.data', 'rb') as f:
    class_names = pickle.load(f)

correct_glioma_tumor = 0
count_glioma_tumor = 0
directory = 'Testing/glioma_tumor'
for filename in os.listdir(directory):
    count_glioma_tumor += 1
    f = os.path.join(directory, filename)
    im = cv2.imread(f)
    img_array = utils.img_to_array(im)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    if "glioma_tumor" == class_names[prediction]:
        correct_glioma_tumor += 1
print(f"GLIOMA TUMOR - Number of test images: {count_glioma_tumor}, Correct: {correct_glioma_tumor}, % correct: {correct_glioma_tumor/count_glioma_tumor * 100:.0f}%")

correct_meningioma_tumor = 0
count_meningioma_tumor = 0
directory = 'Testing/meningioma_tumor'
for filename in os.listdir(directory):
    count_meningioma_tumor += 1
    f = os.path.join(directory, filename)
    im = cv2.imread(f)
    img_array = utils.img_to_array(im)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    if "meningioma_tumor" == class_names[prediction]:
        correct_meningioma_tumor += 1
print(f"MENINGIOMA TUMOR - Number of test images: {count_meningioma_tumor}, Correct: {correct_meningioma_tumor}, % correct: {correct_meningioma_tumor/count_meningioma_tumor * 100:.0f}%")

correct_pituitary_tumor = 0
count_pituitary_tumor = 0
directory = 'Testing/pituitary_tumor'
for filename in os.listdir(directory):
    count_pituitary_tumor += 1
    f = os.path.join(directory, filename)
    im = cv2.imread(f)
    img_array = utils.img_to_array(im)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    if "pituitary_tumor" == class_names[prediction]:
        correct_pituitary_tumor += 1
print(f"PITUITARY TUMOR - Number of test images: {count_pituitary_tumor}, Correct: {correct_pituitary_tumor}, % correct: {correct_pituitary_tumor/count_pituitary_tumor * 100:.0f}%")

correct_no_tumor = 0
count_no_tumor = 0
directory = 'Testing/no_tumor'
for filename in os.listdir(directory):
    count_no_tumor += 1
    f = os.path.join(directory, filename)
    im = cv2.imread(f)
    img_array = utils.img_to_array(im)
    img_array = img_array[tf.newaxis, ...]
    predictions = model(img_array).numpy()
    predictions = list(predictions[0])
    prediction = predictions.index(max(predictions))
    if "no_tumor" == class_names[prediction]:
        correct_no_tumor += 1
print(f"NO TUMOR - Number of test images: {count_no_tumor}, Correct: {correct_no_tumor}, % correct: {correct_no_tumor/count_no_tumor * 100:.0f}%")

count_total = count_glioma_tumor + count_meningioma_tumor + count_pituitary_tumor + count_no_tumor
correct_total = correct_glioma_tumor + correct_meningioma_tumor + correct_pituitary_tumor + correct_no_tumor
print(f"TOTAL - Number of test images: {count_total}, Correct: {correct_total}, % correct: {correct_total/count_total * 100:.0f}%")
