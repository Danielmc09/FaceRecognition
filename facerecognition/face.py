import cv2  #pip3 install opencv-python

# Frontal face detector For Open Source Computer Vision Library
# Clasificador Preentrenado
file = "frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)

# Habilitar Camara
webcam = cv2.VideoCapture(0)

# Capturar imagen desde la camara
# Convertir captura a escala de grises
while True:
    (_, im) = webcam.read()
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(imGray)

    # Cuadro remarcar rotros identificados
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow('OpenCV', im)

    # Fnaliza ciclo While al presionar tela ESC
    key = cv2.waitKey(10)
    if key == 27:
        break

