import time
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(
    api_key="pcsk_4AKL3D_8e2Zut6CfAGo6oJ3ZvZydtRd4xHaVvzRYcqHRV7fePJ3AzdXfrXK7iS7L62hQHg"
)

# Get python object reference to Index
index = pc.Index("my-index") 

# Query the pinecone index passing a query vector
results = index.query(
    vector=[0.0, 0.0, 0.11],
    top_k=1,
    include_values=True
)

# Print the results from fetch operation
print( results )