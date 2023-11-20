import cv2
import numpy as np
import matplotlib.pyplot as plt

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

# --- Cargo imagen -------------------------------------------------------------------------
img = cv2.imread('monedas.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.figure(), plt.imshow(img), plt.show(block=False)

# --- Paso a escala de grises ---------------------------------------------------------------
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.figure(), plt.imshow(img_gray, cmap='gray'), plt.show(block=False)

# --- FILTRO BLUR (homogeneizar fondo) ------------------------------------------------------
img_blur = cv2.medianBlur(img_gray, 9, 2)
plt.figure(), plt.imshow(img_blur, cmap="gray"), plt.show(block=False)

# --- CANNY --------------------------------------------------------------------------------
img_canny = cv2.Canny(img_blur, 10, 54, apertureSize=3, L2gradient=True) 
imshow(img_canny)

# --- DILATACIÓN -----------------------------------------------------
k = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
imgd = cv2.dilate(img_canny, kernel)
imshow(imgd)

# --- RELLENADO DE HUECOS ----------------------------------------------------------------------
img_sin_huecos = imfillhole(imgd)
imshow(img_sin_huecos)

# --- APERTURA --------------------------------------------------------------------------------
ka = 121
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ka,ka))
img_sin_huecos_ap = cv2.morphologyEx(img_sin_huecos, cv2.MORPH_OPEN, kernel3)
imshow(img_sin_huecos_ap)

# --- Componentes conectadas ------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_sin_huecos_ap)
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)

# --- Clasificación ---------------------------------------------------------------------

# Máscara de la misma forma que la imagen original (servirá para borrar dados)
mask = np.ones_like(img_sin_huecos_ap, dtype=np.uint8)

# Imagen en blanco para dibujar las monedas clasificadas
height, width = img_sin_huecos_ap.shape
imagen_monedas_clasif = np.zeros((height, width, 3), dtype=np.uint8)

RHO_TH = 0.066    # Factor de forma (rho) < CIRCULO

# Contadores de tipos de monedas
count_moneda_10 = 0
count_moneda_50 = 0
count_moneda_1 = 0

# Clasifico en base al factor de forma
for i in range(1, num_labels):

    #--- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)

    rho = area/(perimeter**2)
    flag_circular = rho > RHO_TH

    # Debug
    # --- Muestro por pantalla el resultado -----------------------------------
    # print(f"Objeto {i:2d} --> Circular: {flag_circular} --> {rho}")

    # --- Obtengo las coordenadas del rectángulo --------------------------------
    x, y, w, h = cv2.boundingRect(ext_contours[0])

    # --- Clasifico monedas y elimino dados de la imagen --------------------------------
    if flag_circular:

        # Clasificamos moneda según el área
        if 69000 < area < 80000:
            color = (0, 255, 0)  # Color verde para moneda 10 cent
            count_moneda_10 += 1
        elif 80000 <= area < 110000:
            color = (0, 0, 255)  # Color rojo para moneda 1 peso
            count_moneda_1 += 1
        else:
            color = (255, 0, 0)  # Color azul para moneda 50 cent
            count_moneda_50 += 1

        # Pintamos región de la moneda con el color correspondiente
        cv2.drawContours(imagen_monedas_clasif, [ext_contours[0]], 0, color, -1)
    else:
        # Obtenemos imagen sin dados
        # Rellenamos región correspondiente en la máscara con ceros
        mask[y:y+h, x:x+w] = 0

    # Debug
    # if flag_circular:
    #     plt.figure(), plt.imshow(obj, cmap='gray'), plt.title(f"label {i}"), plt.show()

# --- Imagen sin dados -----------------------------------
imagen_sin_dados = img_sin_huecos_ap * mask
imshow(imagen_sin_dados, title="Imagen sin dados")

# --- Monedas clasificadas -----------------------------------
# Color verde --> Moneda 10 cent
# Color rojo --> Moneda 1 peso
# Color azul --> Moneda 50 cent 
imshow(imagen_monedas_clasif, title="Monedas clasificadas")

print("Cantidad de monedas de 10 centavos:", count_moneda_10)
print("Cantidad de monedas de 50 centavos:", count_moneda_50)
print("Cantidad de monedas de 1 peso:", count_moneda_1)

##################################################
# ------------------- DADOS ----------------------

def imclearborder(f):
    kernel = np.ones((3,3),np.uint8)
    marker = f.copy()
    marker[1:-1,1:-1]=0
    while True:
        tmp=marker.copy()
        marker=cv2.dilate(marker, kernel)
        marker=cv2.min(f, marker)
        difference = cv2.subtract(marker, tmp)
        if cv2.countNonZero(difference) == 0:
            break
    mask=cv2.bitwise_not(marker)
    out=cv2.bitwise_and(f, mask)
    return out


# --- Filtro BLUR -----------------------------------
img_blur2 = cv2.medianBlur(img_gray, 25, 21)

# --- Umbralado inverso (quitar dados del costado) ---------------
imgu =  cv2.threshold(img_blur2, 167, 255, cv2.THRESH_BINARY)[1]
imshow(imgu)

# --- Limpiamos bordes -----------------------------------
imgu_sin_bord = imclearborder(imgu)
imshow(imgu_sin_bord)

# --- DILATACIÓN -----------------------------------------------------
k = 22
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k,k))
imgd2 = cv2.dilate(imgu_sin_bord, kernel)
imshow(imgd2)

# --- Relleno huecos -----------------------------------------------------
img_sin_huecos2 = imfillhole(imgd2)
imshow(img_sin_huecos2)

# --- APERTURA --------------------------------------------------------------------------------
ka = 80
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ka,ka))
img_sin_huecos_ap2 = cv2.morphologyEx(img_sin_huecos2, cv2.MORPH_OPEN, kernel3)
imshow(img_sin_huecos_ap2)

# --- Componentes conectadas ------------------------------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_sin_huecos_ap2)
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)

# --- Clasificación ---------------------------------------------------------------------

# Máscara de la misma forma que la imagen original (servirá para borrar dados)
mask = np.zeros_like(img_sin_huecos_ap2, dtype=np.uint8)

RHO_TH = 0.065    # Factor de forma (rho) > CUADRADO

# Clasifico en base al factor de forma
for i in range(1, num_labels):

    #--- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)

    rho = area/(perimeter**2)
    flag_dado = rho < RHO_TH

    # --- Obtengo las coordenadas del rectángulo --------------------------------
    x, y, w, h = cv2.boundingRect(ext_contours[0])

    # --- Máscara para quedarme solo con dados ----------------------------------
    if flag_dado:
      mask[y:y+h, x:x+w] = 1

    # Debug
    # if flag_dado:
    #   plt.figure(), plt.imshow(obj, cmap='gray'), plt.title(f"label {i}"), plt.show()

# --- Aplico máscara -------------------------------------------
imagen_dados = imgu_sin_bord * mask
imshow(imagen_dados, title="Imagen de los dados")


# --- Clasifico dados por cant de huecos -------------------------------------
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_dados)
plt.figure(), plt.imshow(labels, cmap='gray'), plt.show(block=False)


for i in range(1, num_labels):

    #--- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)

    # --- Calculo cantidad de huecos ------------------------------------------
    all_contours, _ = cv2.findContours(obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    holes = len(all_contours) - 1

    # --- Muestro por pantalla el resultado -----------------------------------
    print(f"Dado {i:2d} --> Número {holes}")
