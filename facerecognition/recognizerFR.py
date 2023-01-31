import cv2 as cv  # pip3 install opencv-python
import os

# import imutils #pip install imutils

bancoImagenes = 'DataFaces'
listaUsuarios = os.listdir(bancoImagenes)
print('Usuarios: ', listaUsuarios)

# Reconocimiento de usuarios
face_recognizer = cv.face.EigenFaceRecognizer_create()

# Cargando modelo entrenado con imagenes de usuarios
face_recognizer.read('modeloEigenFaces.xml')

# Habilitar Camara
webcam = cv.VideoCapture(0)

# Llamamos el archivo de detección de objetos
file = "haarcascade_frontalface_alt2.xml"

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + file)

while True:
    (_, im) = webcam.read()
    imAux = im.copy()
    imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    imAuxG = imGray.copy()

    # Clasificador
    faces = face_cascade.detectMultiScale(imGray, 1.3, 5)

    # Cuadro remarcar rotros identificados
    for (x, y, w, h) in faces:
        # Extrae Rostro y Redimensiona 150x150 pixels
        rostro = imAuxG[y:y + h, x:x + w]
        rostro = cv.resize(rostro, (150, 150), interpolation=cv.INTER_CUBIC)
        userID = face_recognizer.predict(rostro)

        cv.putText(im, '{}'.format(userID), (x, y - 5), 1, 1.3, (255, 255, 0), 1, cv.LINE_AA)

        # EigenFaces
        if userID[1] < 5700:
            cv.putText(im, '{}'.format(listaUsuarios[userID[0]]), (x, y - 25), 2, 1.1, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        else:
            cv.putText(im, "Desconocido", (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv.LINE_AA)
            cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Imagen Camara
    cv.imshow('OpenCV', im)

    # Fnaliza ciclo While al presionar tecla ESC
    key = cv.waitKey(10)
    if key == 27:
        break
