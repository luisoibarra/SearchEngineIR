# Search Engine

Implementación de un motor de búsqueda para la recuperación de información. La interfaz gráfica viene dada por una aplicación de Flutter que consume de una API programada en Python con FastAPI.

## Software

Mínimos:

- Python3
  - FastAPI
  - nltk
  - sklearn

Recomendables:

- Flutter

## Uso

### Motor de búsqueda

En la carpeta raíz del proyecto de Python:

- uvicorn main:app 

Este commando iniciará la construcción del motor de búsqueda el cual puede demorar un rato debido al tiempo necesario para procesar los documentos.

Una vez esté listo se expondrá una API Rest a través de la cual se puede hacer peticiones al motor de búsqueda mediante cualquier cliente.

Para comprobar se puede abrir un navegador y escribir:

- http://localhost:8000/query/?query=QUERY&offset=0

### Interfaz Visual

- Correr el proyecto de flutter en la plataforma deseada. 
- Una vez se tenga la aplicación corriendo configurar la dirección a la cual se le hacen los pedidos, la cual debe apuntar al motor de búsqueda.
- Hacer querys 

## Objetivo

El objetivo principal es mostrar los documentos relevantes de un corpus dado una query.

## TODO

- La aplicación de Flutter no se ha integrado aún con la API
- Algunos problemas con la UI de Flutter
- Agregar corpus
- Hacer informe
- Mejorar el modelo vectorial usando un modelo aternativo
- Añadir expansión de query
- Añadir retroalimentación
- Agregar spelling correction

