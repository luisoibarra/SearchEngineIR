# Search Engine

El proyecto provee de una infraestructura flexible con el objetivo de crear un motor de búsqueda. Además expone una API con la cual se pueden hacer las consultas a este.

## Corpus

El corpus sobre el cual se hace el análisis se encuentra en *search_logic/corpus*. Los archivos serán leídos como texto plano.

## Representación

Los documentos y la query son representados como un diccionario. Las llaves básicas son:

- tokens: Los tokens del texto
- text: El texto crudo del documento
- dir: La dirección del documento

## Pipeline

El flujo de los modelos se maneja mediante Pipes los cuales pasan un diccionario que va siendo anotado
en cada fase pudiendo ser utilizados los datos de fases anteriores.

### Tokenización

Fase general de procesamiento de texto para todos los modelos. Permite:

- Tokenización del texto
- Eliminación de las stopwords
- Stemming

Todos estos procesos modifican la llave *tokens* del documento asociado

### Vectorial

Fase de procesamiento de los documentos para su funcionamiento en el modelo vectorial. Permite:

- Calcular idf.
- Convertir documentos a vectores calculando los pesos por ntf * idf.
- Convertir query a vector con factor de suavizado.
- Calcular la similitud entre los documentos y a query y devolver los resultados rankeados fitrados por un umbral para la similitud.

### Rank SVM

Fase de procesamiento que en la cual se hace el entrenamiento del clasificador SVM. Su procesamiento depende en parte del modelo vectorial.

- Entrenamiento del clasificador SVM
- Cálculo de ranking de documentos dado una query

## API

Expone una API a la cual se le pueden hacer las siguientes conusltas:

- *query/?query=QUERY*: Devuelve una lista ordenada por similitud con la query. Vea el modelo **QueryResult**

## Visual

Para el visual se creó un proyecto de streamlit. Correr  `streamlit run visual.py`

## Test

Correr `pytest` en la consola

## Evaluación

Para evaluar el modelo se creó el script eval_model.py, con el cual se prueba el F1, Precisión y Recobrado de los modelos creados.

## Consideraciones

- Se puede hacer los modelos Booleano y Probabilístico sobre la misma infraestructura e incluso combinarlos.
- Aún hace falta trabajar en el corpus

Query expansion
- Expansion de bigrama? Entrenar cual es la palabra en el texto que mas sale luego de otra, o ordenarla por frecuencia para coger una lista. Enfoque global, enfoque local
- Clustering? Dada una query, buscar documentos cercanos a esta y extraer de estos palabras. Se podria hacer tambien algo como un algoritmo que busque cual palabra podria acercar mas la query a diferentes docuemntos.
- Encoder-decoder? Dada la representacion de una query se calcula en el espacio de documentos los K documentos mas cercanos, luego se coge el centroide de esto y se mueve la query con el, una vez se tenga esa nueva representacion de la query se puede hacer un decoder de esta para hacer la expansion

Feedback
- 