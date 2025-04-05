import pandas as pd
# from main.database import database
import os
from pinecone import ServerlessSpec, Pinecone
import time
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.agents import initialize_agent, Tool
from tqdm.auto import tqdm
from getpass import getpass


class EmbeddingHandler:
    def __init__(self, model_name, api_key):
        self.embedder = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )

    def embed_documents(self, documents):
        return self.embedder.embed_documents(documents)

    def embed_query(self, query):
        return self.embedder.embed_query(query)


class PineconeIndexManager:
    def __init__(self, api_key, index_name, spec, dimension=1536):
        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.spec = spec
        self.dimension = dimension

    def initialize_index(self):
        existing_indexes = [index_info['name'] for index_info in self.pc.list_indexes()]
        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)
        self.pc.create_index(
            self.index_name,
            dimension=self.dimension,
            metric='dotproduct',
            spec=self.spec
        )
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)
        self.index = self.pc.Index(self.index_name)

    def upsert(self, vectors):
        self.index.upsert(vectors=vectors)

    def describe_index(self):
        return self.index.describe_index_stats()


class VectorStoreManager:
    def __init__(self, index, embed_query):
        self.vectorstore = PineconeVectorStore(index, embed_query, text_field="text")

    def search(self, query, k=3):
        return self.vectorstore.similarity_search(query, k=k)


class ConversationalAgent:
    def __init__(self, api_key, vectorstore):
        self.llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name='gpt-4o-mini',
            temperature=0.0
        )
        self.memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )
        self.vectorstore = vectorstore
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.vectorstore.as_retriever()
        )
        self.agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=[Tool(name='Knowledge Base', func=self.qa.run, description='General knowledge queries tool')],
            llm=self.llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=self.memory
        )

    def run(self, query):
        return self.agent(query)


# Usage example
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API key: ")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or getpass("Enter your Pinecone API key: ")
data = pd.read_csv('news_data.csv')
spec = ServerlessSpec(cloud="aws", region="us-east-1")

# Initialize components
embedder = EmbeddingHandler(model_name='text-embedding-ada-002', api_key=OPENAI_API_KEY)
index_manager = PineconeIndexManager(api_key=PINECONE_API_KEY, index_name="doc-index", spec=spec)
index_manager.initialize_index()

# Prepare data and upsert
batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    batch = data.iloc[i:i+batch_size]
    metadatas = [{'title': row['heading'], 'text': row['summary']} for _, row in batch.iterrows()]
    documents = batch['summary'].tolist()
    embeddings = embedder.embed_documents(documents)
    ids = batch['id'].tolist()
    vectors = zip(ids, embeddings, metadatas)
    index_manager.upsert(vectors)

index_manager.describe_index()

# Search and Querying
vector_store_manager = VectorStoreManager(index=index_manager.index, embed_query=embedder.embed_query)
agent = ConversationalAgent(api_key=OPENAI_API_KEY, vectorstore=vector_store_manager)

query = "when was the Gusta established?"
response = agent.run(query)
print(response)
