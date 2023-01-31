import cv2 as cv  # pip3 install opencv-python
import os

# Frontal face detector For Open Source Computer Vision Library
# Clasificador Preentrenado

# Llamamos el archivo de detección de objetos
file = "haarcascade_frontalface_alt2.xml"

# Carpeta Banco de Imágenes
bancoImagenes = 'DataFaces'

# Listamos los repositorios que esten creados
listaUsuarios = os.listdir(bancoImagenes)

# Llamamos el modelo EigenFaceRecognizer
face_recognizer = cv.face.EigenFaceRecognizer_create()

# Abrimos la camara
webcam = cv.VideoCapture(0)

# Clasificamos la información
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + file)

# Creamos una bandera en dado caso de que el rostro exista, terminamos el ciclo while
bandera = True

# Validamos si el modelo entrenado existe
if os.path.exists('modeloLBPHFace.xml'):
    # Leemos el modelo entrenado
    face_recognizer.read('modeloLBPHFace.xml')
    while bandera:
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
                cv.putText(im, "Rostro no encontrado\n Registrate por favor!!", (x, y - 20), 2,
                           0.8, (0, 0, 255), 1, cv.LINE_AA)
                cv.rectangle(im, (x, y), (x + w, y + h), (0, 0, 255), 2)
                bandera = False

        # Imagen Camara
        cv.imshow('OpenCV', im)

        # Finaliza ciclo While al presionar tecla ESC
        key = cv.waitKey(10)
        if key == 27:
            break

while True:
    # Nombre de usuario a ser registrado
    print("Nombre de usuario: ")
    usuario = input()

    # Carpeta - Repositorio de fotos de Usuario
    userPath = bancoImagenes + '/' + usuario

    if not os.path.exists(userPath):
        print('Creado Repositorio de usuario: ', userPath)
        os.makedirs(userPath)
        break
    else:
        print("Ya existe, un repositorio con ese nombre.")

# Contador de rostros identificados en la captura de imagen
cont = 0

# Habilitar Camara
webcam = cv.VideoCapture(0)

# Capturar imagen desde la camara
# Convertir captura a escala de grises
while True:
    (_, im) = webcam.read()
    imAux = im.copy()
    imGray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    # Clasificador
    faces = face_cascade.detectMultiScale(imGray)

    # Cuadro remarcar rotros identificados
    for (x, y, w, h) in faces:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extrae Rostro y Redimensiona 150x150 pixels
        rostro = imAux[y:y + h, x:x + w]
        rostro = cv.resize(rostro, (150, 150), interpolation=cv.INTER_CUBIC)

        rostroID = userPath + '/' + usuario
        cv.imwrite(rostroID + '_{}.jpg'.format(cont), rostro)
        cont = cont + 1

        # Imagen Rostro Recortado
        cv.imshow('Rostro', rostro)

        # Imagen Camara
        cv.imshow('OpenCV', im)

    # Finaliza ciclo While al presionar tecla ESC o cuando se guardan 400 imagenes
    key = cv.waitKey(10)
    if key == 27 or cont >= 400:
        break
