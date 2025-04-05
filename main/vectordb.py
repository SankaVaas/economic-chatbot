import getpass
import os


from pinecone import Pinecone, ServerlessSpec

key = os.environ.get('OPENAI_API_KEY')
print(key)

# if not os.environ.get("PINECONE_API_KEY"):
#     os.environ["PINECONE_API_KEY"] = getpass.getpass("Enter your Pinecone API key: ")

pinecone_api_key = os.environ.get('PINECONE_API_KEY')
print(pinecone_api_key)

