import cv2  #pip3 install opencv-python

# Frontal face detector For Open Source Computer Vision Library
# Clasificador Preentrenado
file = "frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)

# Carpeta Banco de Imágenes
bancoImagenes='DataFaces'

# Nombre de usuario a ser registrado
print("Nombre de usuario: ")
usuario=input()

#Contador de rostros identificados en la captura de imagen
cont=0

# Habilitar Camara
webcam = cv2.VideoCapture(0)

# Capturar imagen desde la camara
# Convertir captura a escala de grises
while True:
    (_, im) = webcam.read()
    imAux = im.copy()
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    #Clasificador
    faces = face_cascade.detectMultiScale(imGray)

    # Cuadro remarcar rotros identificados
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.putText(im,"S: Guardar captura de usuario: "+usuario,(10,20),2,0.5,(255,0,0),1,cv2.LINE_AA)
        
        #Imagen Camara
        cv2.imshow('OpenCV', im)

        key = cv2.waitKey(0)
        if key == ord('s'):
            #Extrae Rostro y Redimensiona 150x150 pixels
            rostro = imAux[y:y+h,x:x+w]
            rostro = cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
            rostroID=bancoImagenes+usuario
            cv2.imwrite(rostroID+'_{}.jpg'.format(cont),rostro)
            cont=cont+1
            #Imagen Rostro Recortado
            cv2.imshow('Rostro',rostro)
        elif key == 27:
            break


    # Fnaliza ciclo While al presionar tela ESC
    key = cv2.waitKey(10)
    if key == 27:
        break

