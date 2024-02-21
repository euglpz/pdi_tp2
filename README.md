TUIA - Procesamiento de Imágenes I - TP2

En este proyecto se realizaron dos ejercicios:
1) Detección y clasificación de Monedas y Dados

Tenemos una imagen de monedas de distinto valor y tamaño, y otra imagen de dados sobre un fondo de intensidad no uniforme.
* Se procesó la imagen de manera de segmentar las monedas y los dados de manera automática.
* Luego, clasificamos los distintos tipos de monedas y realizamos un conteo, de manera automática.
* Por último se determinó el número que presenta cada dado mediante procesamiento automático.
---

2) Detección de patentes
   
Se tienen imágenes de la vista anterior o posterior de diversos vehículos donde se visualizan las correspondientes patentes.

* Se implementó un algoritmo de procesamiento de imágenes que detecta automáticamente las patentes y las segmenta.
* Con otro algoritmo se logró segmentar los caracteres de la patente detectada en el punto anterior.

## Instrucciones para correr el proyecto:

1) Descargar el proyecto desde <> Code -> Download ZIP
2) Extraerlo en su PC (la carpeta se llamará *pdi_tp2-main*)
3) Abrir VS Code -> File -> Open Folder... (pdi_tp2-main)
4) Abrimos una nueva terminal
5) Creamos un venv: python -m venv venv
6) Lo activamos: .\venv\Scripts\activate
8) Y dentro de él instalamos los requerimientos: pip install -r .\requirements.txt
9) Nos movemos a la carpeta src: cd src
10) Corremos python: python
11) Y por último abrimos el ejercicio 1 (tp2_ejer1.py) o el 2 (tp2_ejer2.py)
