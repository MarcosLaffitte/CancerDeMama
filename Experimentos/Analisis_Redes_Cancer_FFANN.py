####################################################################################################################
#                                                                                                                  #
# - Universidad Nacional Autonoma de Mexico                                                                        #
# - Centro de Fisica Aplicada y Tecnologia Avanzada                                                                #
# - Licenciatura en Tecnologia                                                                                     #
# - Tesista: Marcos Emmanuel Gonzalez Laffitte                                                                     #
# - Asesor: Dra. Maribel Hernandez Rosales, Insituto de Matematicas UNAM, Juriquilla                               #
#                                                                                                                  #
####################################################################################################################
# CODE #############################################################################################################

# DEPENDENCIES #####################################################################################################
"""No incluidas en Python 3.5"""
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

"""Ya incluidas en python 3.5"""
import pickle
import random
from sys import argv
import warnings
warnings.simplefilter("ignore")
# descomentar esta opcion si se ejecuta en servidor
#plt.switch_backend('agg')

# GLOBAL VARIABLES #################################################################################################
ensayos = 500
redesTodas = []
totalRedes = 0
cuenta = 0
porcentaje = 0
margen = range(50, 100, 10)
neuronas = range(5, 30, 5)
X_entrenamiento = []
Y_entrenamiento = []
X_prueba = []
Y_prueba = []
propiedades = []
etiquetas = []
etiqueta = 0
exactitudMargen = dict()
exactitudNeuronas = dict()
exactitud = 0
models = []
resultAv = []
resultStd = []
totResults = []
save = []
generadas = 0

# MAIN #############################################################################################################

print("\n\n")

# obtener nombre de archivo ----------------------------------------------------------------------------------------
nombre = argv[1]
# abrir archivo de redes de cancer de mama
archivo = open(nombre, "rb")
redesTodas = pickle.load(archivo)
archivo.close()
totalRedes = len(redesTodas)

# Experimento 1:  variando porcentaje de datos ---------------------------------------------------------------------
generadas = 0
for i in range(len(margen)):
    exactitudMargen[margen[i]] = []
    for j in range(ensayos):
        # repartir datos aleatoriamente
        cuenta = 0
        while(redesTodas):
            # obtener red aleatoriamente
            random.shuffle(redesTodas)
            red = redesTodas.pop()
            save.append(red)
            # calcuar porcentaje
            cuenta = cuenta + 1
            porcentaje = (cuenta * 100) / totalRedes
            # crear par de entrenamiento o prueba
            if(porcentaje <= margen[i]):
                propiedades = [float(value) for value in red[0]]
                etiquetas = red[1]
                etiqueta = etiquetas.index("1")
                X_entrenamiento.append(propiedades)
                Y_entrenamiento.append(etiqueta)
            elif(len(X_prueba) < 15):
                propiedades = [float(value) for value in red[0]]
                etiquetas = red[1]
                etiqueta = etiquetas.index("1")
                X_prueba.append(propiedades)
                Y_prueba.append(etiqueta)
            else:
                continue
        # recuperar redes
        redesTodas = [someNetwork for someNetwork in save]
        save = []
        # crear modelo
        clf = MLPClassifier(solver = "sgd", activation = "logistic", hidden_layer_sizes=(5), momentum = 0.1)
        clf.fit(X_entrenamiento, Y_entrenamiento)
        prediction = list(clf.predict(X_prueba))
        exactitud = accuracy_score(Y_prueba, prediction, normalize=True)
        exactitudMargen[margen[i]].append(exactitud)
        # imprimir avance
        generadas = generadas + 1
        print("- avance porcenaje :\t" + str(round(generadas * 100 / (len(margen) * ensayos), 2)) + " %  ...    ", end = "\r")
print("- avance porcenaje : Finalizado             ")
# normalize and plot results
models = []
resultAv = []
resultStd = []
totResults = []
for i in range(len(margen)):
    models.append(margen[i])
    resultAv.append(np.mean(exactitudMargen[margen[i]]))
    resultStd.append(np.std(exactitudMargen[margen[i]]))
    totResults = totResults + exactitudMargen[margen[i]]
someFile = open("DatosPorcentaje.pickle", "wb")
pickle.dump((models, resultAv, resultStd, totResults), someFile)
someFile.close()
exactitudMargen = dict()
plt.plot(models, resultAv, ":o", color = "r", linewidth = 0.5, markersize = 1)
plt.savefig("VariandoPorcentajeNoStd.pdf", dpi = 650)
plt.close()
plt.errorbar(models, resultAv, yerr = resultStd , color = "r", fmt = ":o", capsize = 2, elinewidth = 0.05, capthick = 0.3,
             linewidth = 0.5, markersize = 1)
plt.savefig("VariandoPorcentajeWithStd.pdf", dpi = 650)
plt.close()
models = []
resultAv = []
resultStd = []
totResults = []

print("\n")

# Experimento 2:  variando neuronas en la capa oculta ---------------------------------------------------------------------
generadas = 0
for i in range(len(neuronas)):
    exactitudNeuronas[neuronas[i]] = []
    for j in range(ensayos):
        # repartir datos aleatoriamente
        cuenta = 0
        while(redesTodas):
            # obtener red aleatoriamente
            random.shuffle(redesTodas)
            red = redesTodas.pop()
            save.append(red)
            # calcuar porcentaje
            cuenta = cuenta + 1
            porcentaje = (cuenta * 100) / totalRedes
            # crear par de entrenamiento o prueba
            if(porcentaje <= 90):
                propiedades = [float(value) for value in red[0]]
                etiquetas = red[1]
                etiqueta = etiquetas.index("1")
                X_entrenamiento.append(propiedades)
                Y_entrenamiento.append(etiqueta)
            elif(len(X_prueba) < 15):
                propiedades = [float(value) for value in red[0]]
                etiquetas = red[1]
                etiqueta = etiquetas.index("1")
                X_prueba.append(propiedades)
                Y_prueba.append(etiqueta)
            else:
                continue
        # recuperar redes
        redesTodas = [someNetwork for someNetwork in save]
        save = []
        # crear modelo
        clf = MLPClassifier(solver = "sgd", activation = "logistic", hidden_layer_sizes=(neuronas[i]), momentum = 0.1)
        clf.fit(X_entrenamiento, Y_entrenamiento)
        prediction = list(clf.predict(X_prueba))
        exactitud = accuracy_score(Y_prueba, prediction, normalize=True)
        exactitudNeuronas[neuronas[i]].append(exactitud)
        # imprimir avance
        generadas = generadas + 1
        print("- avance neuronas :\t" + str(round(generadas * 100 / (len(neuronas) * ensayos), 2)) + " %  ...    ", end = "\r")
print("- avance neuronas : Finalizado             ")
# normalize and plot results
models = []
resultAv = []
resultStd = []
for i in range(len(neuronas)):
    models.append(neuronas[i])
    resultAv.append(np.mean(exactitudNeuronas[neuronas[i]]))
    resultStd.append(np.std(exactitudNeuronas[neuronas[i]]))
someFile = open("DatosNeuronas.pickle", "wb")
pickle.dump((models, resultAv, resultStd), someFile)
someFile.close()
exactitudNeuronas = dict()
plt.plot(models, resultAv, ":o", color = "b", linewidth = 0.5, markersize = 1)
plt.savefig("VariandoNeuronasNoStd.pdf", dpi = 650)
plt.close()
plt.errorbar(models, resultAv, yerr = resultStd , color = "b", fmt = ":o", capsize = 2, elinewidth = 0.05, capthick = 0.3,
             linewidth = 0.5, markersize = 1)
plt.savefig("VariandoNeuronasWithStd.pdf", dpi = 650)
plt.close()

print("\n\n")

# END ##############################################################################################################
####################################################################################################################

