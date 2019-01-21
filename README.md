> Aprendizaje automático con propiedades de redes de regulación genética en cáncer de mama

> hecho por: Marcos Laffitte - como parte de su tesis para Licenciatura en Tecnología - UNAM - CFATA


I) Pipeline:


1. Obtener y limpiar redes de cáncer de mama de la base de datos pública The Cancer Network Galaxy (http://tcng.hgc.jp/):

1.1 dicts - /Datos_crudos/RedesCancerMama

1.2 comando - python3.5 LimpiadorDeRedes.py \*.txt

1.3 output - redesCancerMama.pickle


2. Generar redes aleatorias y obtener pares de entrenamiento:

2.1 dicts - /Datos_crudos/GeneradorDePares

2.2 comando - python3.5 GeneradorDePares.py redesCancerMama.pickle

2.3 output - datosRedes.pickle


3. Realizar entrenamiento de redes neuronales aritificales:

3.1 dicts - /Experimentos

3.2 comando - python3.5 Analisis_Redes_Cancer_FFANN.py datosRedes.pickle

3.3 output - DatosNeuronas.pickle, DatosPorcentaje.pickle, VariandoNeuronasNoStd.pdf, VariandoNeuronasWithStd.pdf, VariandoPorcentajeNoStd.pdf, VariandoPorcentajeWithStd.pdf


II) Visualizar pares de entrenamiento:

dicts - /Lista_de_pares_de_entrenamiento/ParesEntrenamiento.txt

comando - python3.5 VisualizarPares.py datosRedes.pickle

output - ParesEntrenamiento.txt
