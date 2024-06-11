from langchain.agents import  initialize_agent, AgentType
from langchain_core.messages import HumanMessage
# need to install youtube-transcript-api
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import CharacterTextSplitter
# need to install faiss package
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
from aiutils.GeminiModel import GeminiModel
from aiutils.utils import OSUtils  
from pydantic import BaseModel, field_validator
import sys
from typing import List

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment")

embeddings = OpenAIEmbeddings(api_key=api_key)
model = GeminiModel()

class Chunks(BaseModel):
    chunks: List[str]

    @field_validator("chunks")
    def check_chunks(cls, v):
        if len(v) == 0:
            raise ValueError("chunks cannot be empty")
        return v
    
    def get_json(self):
        return self.json()
    
    @staticmethod
    def to_instance(self, string: str):
        return Chunks.model_validate_json(string)

def create_vector_db_from_video(video_url: str):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    transcript = loader.load() # returns list of documents
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(transcript[0].page_content)
    vectorStore = FAISS.from_texts(chunks, embeddings)
    return vectorStore

def get_response_from_query(db: FAISS, query: str, num_docs=3):
    docs = db.similarity_search(query, num_docs)
    docs_page_content = "\n\n".join([doc.page_content for doc in docs])
    return docs_page_content

prompt = ChatPromptTemplate.from_messages(
    [
        ("user", """Act as a helpful assistant who can help others learn. Given a youtube transcript excerpt, answer questions about it and other queries. Here is a portion of the transcript:\n\n
        {transcript_excerpt}"""),
        ("user", "Here is the query: {query}"),
    ]
)


print("creativing vector db...")
db = create_vector_db_from_video("https://www.youtube.com/watch?v=OrliU0e09io")
print(db.index.ntotal)
print("getting doc content...")
doc_content = get_response_from_query(db, "Why did React query get so popular?")
print(doc_content)
print("asking model\n\n")

chain = prompt | model.model | model.string_parser
response = chain.invoke({"transcript_excerpt": doc_content, "query": "Why did React query get so popular?"})
print(response)
