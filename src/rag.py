from aiutils.GeminiModel import OpenAIModel
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def load_documents():
    DATA_PATH = "documents"
    loader = DirectoryLoader(DATA_PATH, glob="*.txt")
    documents = loader.load()
    return documents

def create_chroma_db(docs: list[Document], embeddings: OpenAIEmbeddings):
    CHROMA_PATH = "chroma"
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    return db
    

def split_docs(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks

def perform_RAG(query : str, model: OpenAIModel, db: Chroma):
    results = db.similarity_search(query, k=5)
    docs_page_content = "\n\n".join([doc.page_content for doc in results])
    invokeChain = model.generateChatPrompt(f"""Answer any questions the user has about FFMPEG using the provided context and your own knowledge.
                                           Here is the context: {docs_page_content}""")
    response = invokeChain([
        HumanMessage(content=query)
    ])
    return response
    
documents = load_documents()
chunks = split_docs(documents)

model = OpenAIModel()
db = create_chroma_db(chunks, model.embeddings)

response = perform_RAG("WHat is the best way to compress a video while ensuring good quality?", model, db)
print(response)

