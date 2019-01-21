###########################################################################################
#                                                                                         #
# - Universidad Nacional Autonoma de Mexico                                               #
# - Centro de Fisica Aplicada y Tcnologia avanzada                                        #
# - Marcos Emmanuel Gonzalez Laffitte                                                     #
# - Generador de redes aleatorias scale free                                              #
#                                                                                         #
###########################################################################################
# CODE ####################################################################################

# Dependencies ############################################################################
"""Not built in Python 3.5"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpmath import mp, mpf
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.metrics.cluster import normalized_mutual_info_score

"""Built in python 3.5"""
import multiprocessing
from itertools import product
import pickle
from sys import argv
import math
import time
import argparse
import warnings
warnings.simplefilter("ignore")
# uncomment this option if running on server
#plt.switch_backend('agg')

# Large Decimal Numbers Management ###########################################################################################
decimals = 50
mp.dps = decimals
margin = mpf("1e-" + str(decimals))
# ex: y = "2.0005", mpf(y) = 2.0005
# apply only to string-float conversion and operations

# VARIABLES GLOBALES ######################################################################
archivo = None
redes = None

# FUNCTIONS ###############################################################################
# function: obtain dependencies ----------------------------------------------------------------------------------------------
def obtainDependenceValues(graph, interactions, sampleSpace, margin, centrality):
    # local variables
    v1 = ""
    v2 = ""
    theSize = 0
    assigned = False
    listOfDependenceValues = []
    theOrderedDependenceValues = []
    theDependenceValues = dict()
    theInteractionsByDependence = dict()
    centralities = dict()
    termAux1 = 0
    termAux2 = 0
    maxCentrality = 0
    # obtain centralities if requiered
    if(centrality):
        centralities = nx.edge_betweenness_centrality(graph)
        if(not len(set(centralities.values())) == 1):
            maxCentrality = max(list(centralities.values()))
            for (v1, v2) in list(centralities.keys()):
                graph[v1][v2]["force"] = mpf(maxCentrality) - mpf(centralities[(v1, v2)])
    # ontain normalizer
    theSize = graph.size(weight = "force")
    # obtain dependence values
    for (v1, v2) in sampleSpace:
        if(not graph.has_edge(v1, v2)):
            theDependenceValues[(v1, v2)] = mpf(-1) * mpf(graph.degree(v1, weight = "force")) * mpf(graph.degree(v2, weight = "force")) / mpf(4 * theSize * theSize)
        else:
            termAux1 = mpf(graph[v1][v2]["force"]) / mpf(2 * theSize)
            termAux2 = mpf(graph.degree(v1, weight = "force")) * mpf(graph.degree(v2, weight = "force")) / mpf(4 * theSize * theSize)
            theDependenceValues[(v1, v2)] = mpf(termAux1) - mpf(termAux2)
        # save value for interactions, with margin to  evaluate similarity
        if((v1, v2) in interactions):
            assigned = False
            listOfDependenceValues = list(theInteractionsByDependence.keys())
            for i in range(len(listOfDependenceValues)):
                dep = listOfDependenceValues[i]
                if(abs(mpf(theDependenceValues[(v1, v2)]) - mpf(dep)) < margin):
                    theInteractionsByDependence[dep].append((v1, v2))
                    assigned = True
                    break
            if(not assigned):
                theInteractionsByDependence[theDependenceValues[(v1, v2)]] = [(v1, v2)]
    # get ordered dependence values
    theOrderedDependenceValues = list(theInteractionsByDependence.keys())
    theOrderedDependenceValues.sort()
    # end of function
    return(theDependenceValues, theOrderedDependenceValues, theInteractionsByDependence)

# function: obtain d-clusters ------------------------------------------------------------------------------------------------
def obtainClusters(clusterizableGraph):
    # local variables
    theClustList = 0
    # get list of d-clusters 
    theClustList = list(nx.connected_components(clusterizableGraph))
    # end of function
    return(theClustList)

# function: obtain modularity ------------------------------------------------------------------------------------------------
def obtainModularity(someClusters, dependenceValues, margin):
    # local variables
    v1 = ""
    v2 = ""
    theModularity = 0
    clust = None
    # obtain modularity of the partition into clusters
    for clust in someClusters:
        for (v1, v2) in list(product(clust, repeat = 2)):
            theModularity = mpf(theModularity) + mpf(dependenceValues[(v1, v2)])
    # avoid minus cero (-0.000000000000000...)
    if(abs(theModularity) <= margin):
        theModularity = 0
    # end of function
    return(theModularity)

# function: obtain modular image of graph ------------------------------------------------------------------------------------
def obtainModularImage(vertices, interactions, dependenceValues, orderedDependenceValues, interactionsByDependence,  margin):
    # local variables
    graphAux = None
    graphWithHighModularity = None
    interactionsToErase = []
    interactionsAux = []
    communitiesAux = []
    modArr = []
    depArr = []
    modAux = 0
    modMax = 0
    depMin = 0
    done = 0
    goal = 0
    # get total number of edges in the graph
    goal = len(interactions)
    # initialize auxilarity graphs 
    graphAux = nx.Graph()
    graphWithHighModularity = nx.Graph()
    graphWithHighModularity.add_nodes_from(vertices)
    graphWithHighModularity.add_edges_from(interactions)
    graphAux.add_nodes_from(vertices)
    graphAux.add_edges_from(interactions)
    # obtain modularity for graph with all interactions
    communitiesAux = obtainClusters(graphAux)
    modAux = obtainModularity(communitiesAux, dependenceValues, margin)
    modMax = modAux
    depMin = orderedDependenceValues[0]
    modArr.append(round(modAux, 10))
    depArr.append(round(depMin, 10))
    # maximize modularity
    for i in range(len(orderedDependenceValues)):
        # definir interacciones a borrar
        interactionsToErase = interactionsToErase + interactionsByDependence[orderedDependenceValues[i]]
        # reinitialize auxiliary graph
        graphAux = nx.Graph()
        graphAux.add_nodes_from(vertices)
        graphAux.add_edges_from(interactions)
        # errase interactions
        graphAux.remove_edges_from(interactionsToErase)
        if(not (graphAux.size()) == 0):
            # obtain clusters
            communitiesAux = obtainClusters(graphAux)
            # obtain modularity
            modAux = obtainModularity(communitiesAux, dependenceValues, margin)
            modArr.append(round(modAux, 10))
            depArr.append(round(orderedDependenceValues[i + 1], 10))
            # determine if better parition
            if(modAux >= modMax):
                modMax = modAux
                interactionsAux = list(graphAux.edges())
                graphWithHighModularity = nx.Graph()
                graphWithHighModularity.add_nodes_from(vertices)
                graphWithHighModularity.add_edges_from(interactionsAux)
                depMin = round(orderedDependenceValues[i + 1], 10)
    # end of function
    return(obtainClusters(graphWithHighModularity), modMax, depMin)

# function: communty detection by dependence comparison ----------------------------------------------------------------------
def MODC(graph, centrality):
    # global variables
    global margin
    # local variables
    vertices = []
    sampleSpace = []
    interactions = []
    modularImage = None
    modularity = 0
    minDependence = 0
    communities = []
    orderedDepValues = []
    depValues = dict()
    interactionsByDep = dict()
    communitiesDict = dict()
    # get interactions, vertices and pairs
    vertices = list(graph.nodes())
    sampleSpace = list(product(vertices, repeat = 2))
    interactions = list(graph.edges())
    # obtain dependence values
    (depValues, orderedDepValues, interactionsByDep) = obtainDependenceValues(graph, interactions, sampleSpace, margin, centrality)
    # optimize modularity
    (communities, modularity, minDependence) = obtainModularImage(vertices, interactions, depValues, orderedDepValues, interactionsByDep,  margin)
    # fin de funcion
    return(modularity, minDependence, len(communities))
    
# funcion: obtener valores -----------------------------------------------------------------------------------------
def obtenerValores(dirigido, noDirigido):
    # variables locales
    datos = []
    m = 0
    c = 0
    dm = 0
    com = 0
    # 1; orden - ambas
    #print("orden")
    datos.append(str(dirigido.order()))
    # 2; tamaño - dirigida
    #print("tamaño")
    datos.append(str(dirigido.size()))
    # 3; densidad, dirigida
    #print("densidad")
    datos.append(str(nx.density(dirigido)))
    # 4; grado promedio - dirigido
    #print("grado promedio")
    datos.append(str((dirigido.size())/(dirigido.order())))
    # 5; diametro - no dirigido
    #print("diametro")
    datos.append(str(nx.diameter(noDirigido)))
    # 6; radio - no dirigido
    #print("radio")
    datos.append(str(nx.radius(noDirigido)))
    # 7; tamaño de clique mas grande - no dirigida
    #print("clique mas grande")
    datos.append(str(nx.graph_clique_number(noDirigido)))
    # 8; numero de cliques maximales - no dirigida
    #print("cliques maximales")
    datos.append(str(nx.graph_number_of_cliques(noDirigido)))
    # 9; global reaching centrality - dirigido
    #print("reachability")
    datos.append(str(nx.global_reaching_centrality(dirigido)))
    # 10; clustering coefficient - dirigida
    #print("clustering")
    datos.append(str(nx.average_clustering(dirigido)))
    # 11; transitividad - dirigida
    #print("transitivity")
    datos.append(str(nx.transitivity(dirigido)))
    # 12; 13; 14; datos MODC: modularidad, dependencia minima, total de comunidades - no dirigido
    #print("MODC")
    (m, dm, com) = MODC(noDirigido, True)
    datos.append(str(m))
    datos.append(str(dm))
    datos.append(str(com))
    # fin de funcion
    return(datos)

# funcion: generar redes aleatorias y analizar las tres redes ------------------------------------------------------
def ObtenerParesDeDatos(informacion):
    # descomponer entrada
    red_P = informacion[0]
    recipientes = informacion[1]
    infoRedes_P = recipientes[0]
    cuenta_P = recipientes[1]
    total_P = recipientes[2]
    # variables locales
    orden = 0
    aristas = 0
    inDegreeSequence = []
    outDegreeSequence = []
    conexa = False
    aleatoriaGNM = nx.DiGraph()
    aleatoriaHH = nx.DiGraph()
    datosRed = []
    grafo = nx.DiGraph
    grafoNoDirigido = nx.Graph()
    # obtener datos de la real
    orden = red_P.order()
    aristas = red_P.size()
    inDegreeSequence = [deg for node, deg in red_P.in_degree()] 
    outDegreeSequence = [deg for node, deg in red_P.out_degree()]
    grafoNoDirigido = nx.Graph(red_P)
    datosRed = obtenerValores(red_P, grafoNoDirigido)
    infoRedes_P.append((datosRed, ["1", "0"]))
    # generar gnm preservando orden y tamaño
    conexa = False
    while(not conexa):
        aleatoriaGNM = nx.gnm_random_graph(orden, aristas, directed = True)
        grafoNoDirigido = nx.Graph(aleatoriaGNM)
        if(nx.is_connected(grafoNoDirigido)):
            conexa = True
    datosRed = obtenerValores(aleatoriaGNM, grafoNoDirigido)
    infoRedes_P.append((datosRed, ["0", "1"]))
    # generar havel hakimi para preservar secuencias de grado
    conexa = False
    while(not conexa):
        aleatoriaHH = nx.directed_havel_hakimi_graph(inDegreeSequence, outDegreeSequence)
        grafoNoDirigido = nx.Graph(aleatoriaHH)
        if(nx.is_connected(grafoNoDirigido)):
            conexa = True        
    datosRed = obtenerValores(aleatoriaHH, grafoNoDirigido)
    infoRedes_P.append((datosRed, ["0", "1"]))
    # imprimir avance
    cuenta_P.value = cuenta_P.value + 1
    print(" - avance: " + str(round((cuenta_P.value * 100) / total_P.value, 2 )) + " % ...", end = "\r")

# MAIN ####################################################################################
print("\n\n")

# abrir archivo con los grafos ------------------------------------------------------------
archivo = open(argv[1], "rb")
redes = pickle.load(archivo)
archivo.close()

# iniciar analisis aleatorio de las redes -------------------------------------------------
if(__name__ == "__main__"):
    # preparar datos
    myMan = multiprocessing.Manager()
    infoRedes = myMan.list()
    cuenta = myMan.Value("i", 0)
    total = myMan.Value("i", len(redes))
    infoRedesMulti = [(infoRedes, cuenta, total)] * len(redes)
    parejasDatos = list(zip(redes, infoRedesMulti))
    # preparar procesadores
    procesadores = math.floor(int(multiprocessing.cpu_count()) * 0.9)
    if(procesadores == 0):
        procesadores = 1
    thePool = multiprocessing.Pool(processes = procesadores)
    # analizar redes en paralelo
    thePool.map(ObtenerParesDeDatos, parejasDatos)
    thePool.close()

# guardar redes ---------------------------------------------------------------------------
archivo = open("datosRedes.pickle", "wb")
pickle.dump(list(infoRedes), archivo)
archivo.close()
print("> Finalizado                                                                  \n\n")

# END #####################################################################################
###########################################################################################
