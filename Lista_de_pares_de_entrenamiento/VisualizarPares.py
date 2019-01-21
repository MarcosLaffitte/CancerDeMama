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

import pickle
from sys import argv
import warnings
warnings.simplefilter("ignore")
# descomentar esta opcion si se ejecuta en servidor
#plt.switch_backend('agg')

# cargar archivo de datos
someFile = open(argv[1], "rb")
pares = pickle.load(someFile)
someFile.close()

# imprimir 144 pares de entrenamiento
someFile = open("ParesEntrenamiento.txt", "w")

someFile.write("# Total de pares: \t" + str(len(pares)) + "\n")
someFile.write("# orden\ttamaño\tdensidad\tgrado-promedio\tdiametro\tradio\ttamaño-clique-mas-grande\tcliques-maximales\talcance-global\tagrupamiento\ttransitividad\tmodularidad\tdependencia\tcomunidades\t>\t[1, 0] = real ; [0, 1] = aleatoria\n")

for par in pares:
    a = "\t".join(par[0])
    b = "\t".join(par[1])
    someFile.write(a + "\t>\t" + b + "\n")

someFile.close()
    
# END ##############################################################################################################
####################################################################################################################

