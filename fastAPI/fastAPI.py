import uvicorn
from fastapi import FastAPI, Request, Form, File, UploadFile, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
from pydantic import BaseModel

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)

from workflow import allSteps, allStepsPDF

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from loggerLog import *

app = FastAPI()
templates = Jinja2Templates("fastAPI/templates")
app.mount("/static", StaticFiles(directory="fastAPI/static"), name="static")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    exc_str = f'{exc}'.replace('\n', ' ').replace('   ', ' ')
    logger.error(exc_str)
    content = {'status_code': 10422, 'message': exc_str, 'data': None}
    return JSONResponse(content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY)


# presents a html form to input data to generate questions using plain text or a pdf file
# inputs include text or pdf file, and answer selection, question generation and distractor selection methods
# answer selection, question generation and distractor selection methods are by default the methods with which we got better results
@app.get('/form/', response_class=HTMLResponse)
async def getForm(request: Request):
    logger.info("Returning html form")
    return templates.TemplateResponse("form.html", {"request": request})


# POST method to get questions generated using plain text as the source
@app.post('/result/', response_class=HTMLResponse)
async def postForm(request: Request, text: str = Form(...), answerSelection: str = Form(...), questionGeneration: str = Form(...), distractorSelection: str = Form(...), maxNumberQuestions: str = Form(...)):
    try:
        logger.info("Calling main question generation function")
        result = allSteps(text, answerSelection, questionGeneration, distractorSelection, maxNumberQuestions)
    except Exception as e:
        logger.error("Exception in main question generation function: %s", e)
    
    logger.info("Response to text form")
    return templates.TemplateResponse("data.html", {"request": request, 'result': result})


# POST method to get questions generated using pdf files
# the pdf file is "uploaded" (copied to this project directory)
# after generating the questions, the file is deleted 
@app.post("/uploadfile/")
async def create_file(request: Request, file: UploadFile = File(...), answerSelection: str = Form(...), questionGeneration: str = Form(...), distractorSelection: str = Form(...), maxNumberQuestions: str = Form(...)):

    if file.filename:
        try:
            logger.info("Uploading File: %s" % file.filename)
            with open(f'{file.filename}', "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error("Exception uploading file ", file.filename, ": ", e)
    else:
        logger.error('Empty filename string')

    try:
        logger.info("Calling main question generation function")
        result = allStepsPDF(file.filename, answerSelection, questionGeneration, distractorSelection, maxNumberQuestions)
    except Exception as e:
        logger.error("Exception in main question generation function: %s", e)

    try:
        logger.info("Removing uploaded file: %s", file.filename)
        os.remove(file.filename)
    except Exception as e:
        print("Exception removing uploaded file: %s", e)
        
    logger.info("Response to file form")
    return templates.TemplateResponse("data.html", {"request": request, 'result': result})


# structure of an item obtained with json used in POST method "/postJson/"
class Item(BaseModel):
    text: str
    answerSelection: str
    questionGeneration: str
    distractorSelection: str
    maxNumberQuestions: str
    
jsonItem = {}

# using a json as the input, we obtain an item with the structure presented above
# the function use to generate questions is the same to the POST method result
@app.post("/postJson/")
async def createItem(item: Item):
    logger.info("Creating item with JSON input")
    jsonItem = item
    result = {}

    if jsonItem.text and jsonItem.answerSelection and jsonItem.questionGeneration and jsonItem.distractorSelection and jsonItem.maxNumberQuestions:
        try:
            logger.info("Calling main question generation function")
            result = allSteps(jsonItem.text, jsonItem.answerSelection, jsonItem.questionGeneration, jsonItem.distractorSelection, jsonItem.maxNumberQuestions)
        except Exception as e:
            logger.error("Exception in main question generation function: %s", e)
    else:
        logger.error("Missing input arguments")
    
    return result


if __name__ == '__main__':
    uvicorn.run(app)