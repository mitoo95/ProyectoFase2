Utilizar este modelo se vuelve facil utilizando sentence-transformers:
```
pip install -U sentence-transformers
```

Luego este modelo puede ser utilizado de esta manera:
```
modelEmbedding = SentenceTransformer('espejelomar/sentece-embeddings-BETO')
modelEmbedding.save('ruta donde se quiere guardar el modelo embedding')
embeddings = modelEmbedding.encode(sentences)
```

Correr un virtual environment e instalar paquetes con requirements.txt