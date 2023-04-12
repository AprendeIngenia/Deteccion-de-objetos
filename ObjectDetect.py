# Importamos librerias
import cv2

# Guardamos en variables los pesos, la arquitectura y las clases del modelo previamente entrenado
model = 'frozen_inference_graph.pb'
config = 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt'
clases = 'coco_labels.txt'

# Extraemos las etiquetas del archivo
with open(clases) as cl:
    labels = cl.read().split("\n")
print(labels)

# Leemos el modelo con TensorFlow
net = cv2.dnn.readNetFromTensorflow(model, config)

# Creamos una función de detección
def object_detect(net, img):
    # Dimensiones
    dim = 300

    # Preprocesamos nuestra imagen
    blob = cv2.dnn.blobFromImage(img, 1.0, size = (dim, dim), mean = (0,0,0), swapRB = True, crop = False)

    # Pasamos nuestra imagen preprocesada a la red
    net.setInput(blob)

    # Extraemos los objetos detectados
    objetos = net.forward()

    return objetos

# Creamos una función de mostrar texto
def text(img, text, x, y):
    # Extraemos el tamaño del texto
    sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
    # Extraemos el tamaño
    dim = sizetext[0]
    baseline = sizetext[1]

    # Creamos un rectangulo negro con el tamaño apropiado
    cv2.rectangle(img, (x, y-dim[1] - baseline), (x + dim[0], y + baseline), (0,0,0), cv2.FILLED)
    # Mostramos el texto
    cv2.putText(img, text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)



# Creamos una función de mostrar objetos
def dibujar_objetos(img, objects, umbral = 0.5):

    # Extraemos info
    filas = img.shape[0]
    colum = img.shape[1]

    # Para los objetos detectados
    for i in range(objects.shape[2]):
        # Buscamos su clase y confianza de detección
        clase = int(objects[0, 0, i, 1])
        puntaje = float(objects[0, 0, i, 2])

        # Extraemos sus coordenadas y las normalizamos a pixeles
        x = int(objects[0, 0, i, 3] * colum)
        y = int(objects[0, 0, i, 4] * filas)
        w = int(objects[0, 0, i, 5] * colum - x)
        h = int(objects[0, 0, i, 6] * filas - x)

        # Revisamos si superamos el umbral
        if puntaje > umbral:
            # Mostramos la clase
            text(img, "{}".format(labels[clase]), x, y)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)

# Creamos la Video Captura
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Creamos un ciclo para ejecutar nuestros Frames
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()

    # Realizamos las detecciones
    detect = object_detect(net, frame)

    # Mostramos las detecciones
    dibujar_objetos(frame, detect)

    # Mostramos los Frames
    cv2.imshow("VIDEO CAPTURA", frame)

    # Cerramos con lectura de teclado
    t = cv2.waitKey(1)
    if t == 27:
        break

# Liberamos la VideoCaptura
cap.release()
# Cerramos la ventana
cv2.destroyAllWindows()


