import streamlit
from GeminiModel import GeminiModel
from langchain_core.messages import HumanMessage

streamlit.title("Pets name generator")

animal_type = streamlit.sidebar.selectbox("What is your pet?", (
    "Dog",
    "Cat",
    "Fish",
))

if animal_type:
    animal_color = streamlit.sidebar.text_input(f"What is the color of your {animal_type}?")

model = GeminiModel()
messages = [
    HumanMessage(content='Given a pet type and its color, generate a list of 5 names for the pet. The response should be in JSON format, with a single "names" property set to a string array of the names.'),
]

if animal_color:
    messages.append(HumanMessage(content=f"{animal_type}, {animal_color}"))
    response = model.complete(messages, response_type="json")
    for i, name in enumerate(response["names"]):
        streamlit.write(f"{i}. {name}")