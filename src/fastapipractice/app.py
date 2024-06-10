from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
import json


app = FastAPI()

# basic route
@app.get("/")
def home():
    return {"message": "Hello World"}

# using query strings
@app.get("/search")
def home(q: str):
    return {"message": f"searching for {q}"}
    # /search?q=dogs -> {"message": "searching for dogs"}

# route parameters
@app.get("/dogs/{name}")
def get_dog(name: str):
    # returns error as JSON response with statuscode
    if name == "pitbull":
        raise HTTPException(status_code=404, detail="dOG IS too dangerous!")
    return {"dog_name": name}



# using JSON for request payload
class Item(BaseModel):
    name: str
    price: float
    
    @field_validator("price")
    def validate_price(cls, value):
        if value < 0:
            raise ValueError("Price must be greater than 0")
        return value

class ResponsePayload(Item):
    message: str

# now instead of query string, the parameter will represent the request body
@app.post("/items", response_model=ResponsePayload)
def create_item(item: Item):
    obj = json.loads(item.model_dump_json())
    obj['message'] = "Item created successfully"
    return obj