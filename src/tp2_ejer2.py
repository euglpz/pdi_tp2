import cv2
import numpy as np
import matplotlib.pyplot as plt
import itertools


def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:
        plt.show(block=blocking)

def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection
    return expanded_intersection

def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh

# Función para calcular distancia entre dos puntos
def calcular_distancia(punto1, punto2):
    x1, y1 = punto1
    x2, y2 = punto2
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# Función para encontrar puntos cercanos a los centroides
def puntos_cercanos(centroids, D):
    puntos_cercanos = []

    # Itera sobre todas las combinaciones posibles de dos puntos en el conjunto
    for par_puntos in itertools.combinations(centroids, 2):
        punto1, punto2 = par_puntos
        distancia = calcular_distancia(punto1, punto2)

        # Si la distancia es menor a D, agrega ambos puntos a la lista
        if distancia < D:
            if tuple(punto1) not in puntos_cercanos:
                puntos_cercanos.append(tuple(punto1))
            if tuple(punto2) not in puntos_cercanos:
                puntos_cercanos.append(tuple(punto2))

    return puntos_cercanos

# Función para recortar patentes
def recortar_patente (img_path):
  # --- Cargo imagen -------------------------------------------------------------------------
  img = cv2.imread(img_path, cv2.IMREAD_COLOR)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # --- Paso a escala de grises ---------------------------------------------------------------
  img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

  # --- Umbralado -----------------------------------
  imgu =  cv2.threshold(img_gray, 135, 255, cv2.THRESH_BINARY)[1]
  #imshow(imgu)

  # --- Componentes conectadas ------------------------------------------------------------
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imgu)
  #plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)

  # --- Clasificación ---------------------------------------------------------------------

  # Imagen en blanco para dibujar las patentes
  height, width = imgu.shape
  patentes = np.zeros((height, width, 1), dtype=np.uint8)

  for i in range(1, num_labels):

      #--- Selecciono el objeto actual -----------------------------------------
      obj = (labels == i).astype(np.uint8)
    
      #--- Busco contornos y calculo área -----------------------------------------
      ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      area = cv2.contourArea(ext_contours[0])

      # Debug
      # --- Muestro por pantalla el resultado -----------------------------------
      #if area > 1:
      #  print(f"Objeto {i:2d} --> {area}")

      # --- Obtengo las coordenadas del rectángulo --------------------------------
      x, y, w, h = cv2.boundingRect(ext_contours[0])
      rel = h/w

      # --- Clasifico por área --------------------------------
      if 10 < area < 100 and 1.4 < rel < 2.5:
        color = (255)  
        #print(f"Objeto{i:2d} Ancho {w} Alto {h}")
      else:
        color = (0)

      # Pintamos región de la patente clasificada por área anteriormente
      cv2.drawContours(patentes, [ext_contours[0]], 0, color, -1)

  # --- Componentes conectadas ------------------------------------------------------------
  num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(patentes)

  # Obtenemos puntos cercanos a los centroides de los caracteres
  pat = puntos_cercanos(centroids[1:], 20)

  # Imagen en blanco
  patentes = np.zeros((height, width, 1), dtype=np.uint8)
  
  # Listas con coordenadas de las regiones de las patentes
  anchos=[]
  altos=[]
  xs=[]
  ys=[]

  for i in range(1, num_labels):
    obj = (labels == i).astype(np.uint8)
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(ext_contours[0])

    if tuple(centroids[i]) in pat:
      color = (255)  
      anchos.append(w)
      altos.append(h)
      xs.append(x)
      ys.append(y)
    else:
        color = (0)  

    # Pintamos región de la patente
    cv2.drawContours(patentes, [ext_contours[0]], 0, color, -1)

    #imshow(patentes, title="Patente")
  
  # Nos quedamos con la regíon de la patente de acuerdo a sus coordenadas y ajustando
  x_inicial=min(xs)-2*max(anchos)
  y_inicial=min(ys)-max(altos)
  x_final=x_inicial+12*(max(anchos))
  y_final=y_inicial+3*max(altos)

  # Recortamos
  recorte_patente = img[y_inicial:y_final, x_inicial:x_final]

  #imshow(recorte_patente, title="Patente")

  return recorte_patente

patente1 = recortar_patente("img01.png")
patente2 = recortar_patente("img02.png")
# patente3 = recortar_patente("img01.png")
patente4 = recortar_patente("img03.png")
patente5 = recortar_patente("img04.png")
# patente6 = recortar_patente("img01.png")
# patente7 = recortar_patente("img01.png")
patente8 = recortar_patente("img08.png")
# patente9 = recortar_patente("img01.png")
patente10 = recortar_patente("img10.png")
patente11 = recortar_patente("img11.png")
patente12 = recortar_patente("img12.png")

imshow(patente1)