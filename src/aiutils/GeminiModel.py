import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from typing import List, Tuple, Dict
import sys


class OpenAIModel:
    string_parser = StrOutputParser()
    json_parser = SimpleJsonOutputParser()
    def __init__(self):
        load_dotenv()
        api_key = os.environ["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment")
        self.model = ChatOpenAI(model="gpt-3.5-turbo", api_key=api_key)
        self.embeddings = OpenAIEmbeddings(api_key=api_key)
    
    def generateChatPrompt(self, prompt: str, response_type="text"):
        prompt = ChatPromptTemplate.from_messages([
            ("system", prompt),
            MessagesPlaceholder("messages")
        ])
        parser = self.string_parser if response_type == "text" else self.json_parser
        chain = prompt | self.model | parser
        def invokeChain(messages: List[HumanMessage | SystemMessage]):
            return chain.invoke({"messages": messages})
        return invokeChain
    
    def generateStructuredPrompt(self, prompt:str, obj: BaseModel):
        parser = JsonOutputParser(pydantic_object=obj)
        prompt = PromptTemplate(
            template=prompt + "\n{format_instructions}\n\n{query}",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.model | parser
        def invokeChain(query: str):
            return chain.invoke({"query": query})
        return invokeChain




class GeminiModel:
    string_parser = StrOutputParser()
    json_parser = SimpleJsonOutputParser()
    def __init__(self, temperature = 0.7, max_output_tokens = None, top_p = None, top_k = None):
        load_dotenv()
        api_key = os.environ["GEMINI_API_KEY"]
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment")
        self.model = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key,
                                            temperature = temperature, max_output_tokens = max_output_tokens, top_p = top_p, top_k = top_k)
        self.vision_model = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=api_key,
                                                   temperature = temperature, max_output_tokens = max_output_tokens, top_p = top_p, top_k = top_k)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", 
            google_api_key=api_key
        )

    
    def _set_chain(
            self, 
            promptTemplate: ChatPromptTemplate, 
            parser: StrOutputParser | SimpleJsonOutputParser | JsonOutputParser | None = None
        ):
        if not parser:
            self.parser = self.string_parser
        else:
            self.parser = parser
        self.prompt_chain = promptTemplate | self.model | self.parser
    
    def complete(self, messages, response_type="text"):
        chain = None
        if response_type == "json":
            chain = self.model | self.json_parser
        else:
            chain = self.model | self.string_parser
        return chain.invoke(messages)
    
    def getEmbedding(self, text):
        return self.embeddings.embed_query("hello, world!")
    
    def getPromptResponse(self, data: dict):
        if not self.prompt_chain:
            raise ValueError("Prompt template is not set")
        return self.prompt_chain.invoke(data)

    def generateStructuredPrompt(self, prompt:str, obj: BaseModel):
        parser = JsonOutputParser(pydantic_object=obj)
        prompt = PromptTemplate(
            template=prompt + "\n{format_instructions}\n\n{query}",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        chain = prompt | self.model | parser
        def invokeChain(query: str):
            return chain.invoke({"query": query})
        return invokeChain
    
    def chatWithImage(self, imageData: str | Image.Image, prompt: str, messages = []):


        """Accepts image data with prompt for completion
        
            Parameters
            ----------
            imageData : str
                the data to image. Can be a URL or base64 encoded image or a PIL Image object or
                a filepath.
            prompt : str
                The text prompt to accompany the text
            messages : list
                The list of messages in the previous conversation
        """
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": prompt,
                },  # You can optionally provide text parts
                {"type": "image_url", "image_url": imageData},
            ]
        )
        return self.vision_model.invoke([*messages, message]).content


class Animal(BaseModel):
    names: List[str] = Field(description="The list of pet names generated")
    
# export only geminimodel and openai model
__all__ = [GeminiModel, OpenAIModel]


if __name__ == "__main__":
    model = GeminiModel()
    messages = [
        HumanMessage(content='Given a pet type and its color, generate a list of 5 names for the pet. The response should be in JSON format, with a single "names" property set to a string array of the names.'),
        HumanMessage(content='Dog, Brown')
    ]
    response = model.complete(messages, response_type="json")
    print(response)

    # prompt = model.generateStructuredPrompt(
    #         prompt="Given a pet type and its color, generate 5 fitting pet names.",
    #         obj=Animal
    #     )
    # print(prompt("Dog, Brown"))