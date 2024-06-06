import numpy as np
import cv2 as cv
import pandas as pd

import os

directorio_fuentes = "./Deformable-Objects-Dataset/Press"
archivos_fuente = {
    "sponge1": "sponge_centre_100.avi"
}
for k, v in archivos_fuente.items():
    archivos_fuente[k] = os.path.join(directorio_fuentes, v)

def extrae_np(archivo, n=-1):
    """
    Dado un archivo genera un arreglo de numpy
    n es el número de cuadros a leer, si n es mayor al número de cuadros
    disponibles o es -1 se devuelven todos los cuadros disponibles.
    """
    cap = cv.VideoCapture(archivo)
    if not cap.isOpened():
        print("No se pudo abrir", archivo)
        return
    ancho = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    num_cuadros = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    if n == -1 or n > num_cuadros:
        n = num_cuadros

    arreglo_datos = np.zeros((num_cuadros, alto, ancho, 6))
    i = -1
    while cap.isOpened():
        ret, cuadro_bgr = cap.read()
        # mientras hay cuadros ret es verdadero
        if not ret:
            break
        i += 1
        if i > n:
            break
        cuadro_hsv = cv.cvtColor(cuadro_bgr, cv.COLOR_BGR2HSV)
        arreglo_datos[i] = np.dstack((cuadro_bgr, cuadro_hsv))
        
    cap.release()
    return arreglo_datos

def crea_arreglo_datos(arreglo_datos_4D, cuadros=0):
    """
    Toma el arreglo 4D y crea una tabla con las características
    originales y las derivadas sólo para los cuadros indicados.
    """
    if type(cuadros) == int:
        cuadros = [cuadros]
    
    dims = arreglo_datos_4D.shape
    num_cuadros = dims[0]
    rens = dims[1]
    cols = dims[2]
    canales = dims[3]
    paso = rens * cols
    num_datos = len(cuadros) * paso       # num_cuadros * ren * col
    num_características = canales + 3       # más cuadro, ren, col
    arreglo_datos = np.zeros((num_datos, num_características))
    #print("Tamaño tabla:", arreglo_datos.shape)

    # Agrega cuadro por cuadro
    ind = 0
    maxj = cols - 1
    for num_cuadro in cuadros:
        cuadro = arreglo_datos_4D[num_cuadro].reshape(-1, canales)
        #print(f"Rens tabla {0}: {1}", num_cuadro , cuadro.shape)
        i = 0
        j = 0
        for renglón in cuadro:
            arreglo_datos[ind,0] = num_cuadro
            arreglo_datos[ind,1:3] = (i, j)
            arreglo_datos[ind,3:] = renglón
            ind += 1
            if j < maxj:
                j += 1
            else:
                i = i + 1
                j = 0
    return arreglo_datos

escenarios = archivos_fuente.keys()
def obtén_tabla_datos(nombre="sponge1", conjuntos=[0, 30, 75]):
    """
    Llama a las funciones que:
    1. Lee el video y guarda las imágenes en un arreglo 4D
    2. Genera las características derivadas y acomoda los datos en renglones
    Con esto crea la tabla de pandas.
    Regresa la tabla de pandas y el arreglo 4D original.
    """
    print("Leyendo numpy...")
    arreglo_datos_4D = extrae_np(archivos_fuente["sponge1"], -1)
    if arreglo_datos_4D is None:
        print("Error al leer los datos de ", nombre)
        return
    print("Organizando renglones...")
    arreglo_datos = crea_arreglo_datos(arreglo_datos_4D, conjuntos)
    print("Creando tabla de datos...")
    tabla_datos = pd.DataFrame(data=arreglo_datos,
                          columns=('f','i','j','B','G','R','H','S','V'))
    print("Terminado")
    return tabla_datos, arreglo_datos_4D

def convierte_a_tabla(arreglo_datos_3D):
    """
    Dada la imagen de un cuadro, la devuelve como tabla de pandas.
    """
    arreglo_datos_4D = arreglo_datos_3D[np.newaxis, ...]
    arreglo_datos = crea_arreglo_datos(arreglo_datos_4D, 0)
    tabla_datos = pd.DataFrame(data=arreglo_datos,
                          columns=('f','i','j','B','G','R','H','S','V'))
    return tabla_datos

def np_a_img(datos3D):
    """
    Recibe la imagen del cuadro con 6 canales y regresa
    una vista RGB
    """
    return datos3D[:,:,[2,1,0]].astype(int)
