Aprendizaje automático con propiedades de redes de regulación genética en cáncer de mama

I) Pipeline:

1. Obtener y limpiar redes de cáncer de mama de la base de datos pública The Cancer Network Galaxy (http://tcng.hgc.jp/): 

1.1 dicts - /Datos_crudos/RedesCancerMama

1.2 commando - python3.5 LimpiadorDeRedes.py \*.txt

2. Generar redes aleatorias y obtener pares de entrenamiento:

dicts - /Datos_crudos/GeneradorDePares

3. Realizar entrenamiento de redes neuronales aritificales:

dicts - /Experimentos

II) Visualizar pares de entrenamiento:
dicts - /Lista_de_pares_de_entrenamiento/ParesEntrenamiento.txt

hecho por: Marcos Laffitte - como parte de su tesis para Licenciatura n Tecnología - UNAM - CFATA
