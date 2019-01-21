#################################################################################
#                                                                               #
# - UNAM - CFATA                                                                #
# - Licenciatura en Tecnología                                                  #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                  #
# - Tutor: Dra. Maribel Hernández Rosales                                       #
# - Programa: Limpiador de redes relacionadas al cancer                         #
# - Recibe: listas de adyacencia originales                                     #
# - Devuelve: listas de adyacencia en dos columnas para enfermas                #
# - version: 1.0                                                                #
# - lenguaje: python                                                            #
# - version de Lenguaje: Python 3.5.3 :: Anaconda custom (64-bit)               #
#                                                                               #
#################################################################################
# codigo ########################################################################


# informacion de dependencias ###################################################
"""
1. sys: integrada en python
2. os: integrada en python
3. networkx: 1.11 - http://networkx.github.io/
"""

# dependencias ##################################################################
import sys
import os
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pickle

# variables globales ############################################################
redes = sys.argv[1:]
red = None
network = nx.DiGraph()
vertices = []
arregloRedes = []
cuenta = 0

# funciones #####################################################################
# funcion: leer red desde archivo -----------------------------------------------
def leerRed(nombre):
    # variables locales
    digrafo = nx.DiGraph()
    archivo = None
    linea = None
    lineaArreglo = []
    padre = ""
    hijo = ""
    edgeScInd = 0
    edgeScore = 0
    # tomar info de archivo
    archivo = open(nombre, "r")
    totLineas = archivo.readlines()
    archivo.close()
    # crear red leyendo archivo
    for linea in totLineas:
        lineaArreglo = linea.split("\t")
        padre = lineaArreglo[0].strip()
        hijo = lineaArreglo[1].strip()
        if(not padre == "Parent"):
            digrafo.add_edge(padre, hijo)
    # fin de la funcion
    return(digrafo)
    
# main ##########################################################################

print("\n\n")
# analizar redes para obtener orden
cuenta = 0
orden = 0
for red in redes:
    network = nx.DiGraph()
    # leer red de archivo
    network = leerRed(red)
    # guardar red solo si su orden es diferente de 8000
    orden = network.order()
    if(not  orden > 500):
        arregloRedes.append(network)
    # imprimir avance
    cuenta = cuenta + 1
    print(" - avance: " + str(round((cuenta * 100) / len(redes), 2)) + "%", end = "\r")
# guardar redes en pickle
archivo = open("redesCancerMama.pickle", "wb")
pickle.dump(arregloRedes, archivo)
archivo.close()
# mensaje
print("Finalizado                                            \n\n")

# fin ###########################################################################
#################################################################################
