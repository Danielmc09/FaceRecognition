import cv2 as cv  # pip3 install opencv-python
import os

# import imutils #pip install imutils
import numpy as np

bancoImagenes = 'DataFaces'

listaUsuarios = os.listdir(bancoImagenes)

labels = []
facesData = []
label = 0

for nameDir in listaUsuarios:
    userPath = bancoImagenes + '/' + nameDir
    print('Cargando imagenes de repositior de imagenes del usuario: ' + nameDir)

    for fileName in os.listdir(userPath):
        print('Rostro: ', fileName)
        labels.append(label)
        facesData.append(cv.imread(userPath + '/' + fileName, 0))
    label = label + 1

# Metodo de entrenamiento Eigenfaces
face_recognizer = cv.face.EigenFaceRecognizer_create()

# Fase de Entrenamiento
print('Entrenaniento en proceso...')
face_recognizer.train(facesData, np.array(labels))
print('Entrenaniento finalizado')

# Almacenamiento del Modelo entrenado
print('Guardando modelo entrenado...')
face_recognizer.write('modeloEigenFaces.xml')
print('Modelo almacenado')
