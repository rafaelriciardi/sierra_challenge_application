import os
import uvicorn
import warnings
from solution import SpamClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

warnings.filterwarnings("ignore")

app = FastAPI()
model = SpamClassifier()

######################## METODOS HTTP ############################

@app.get("/")
async def root():
    return {"message": "Im Alive"}

class ClassificationInput(BaseModel):
    content: str

@app.post("/classificate_email/")
async def classificate_email(input_data: ClassificationInput):
    content = input_data.content

    classification_response = model.predict(email=content)

    return classification_response

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)