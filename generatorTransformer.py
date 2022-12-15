import json
from transformers import file_utils                             #to find where the models are saved locally
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer   #to use the model

import question_generation.pipelines as QGPipelines
nlpQuestion = QGPipelines.pipeline("question-generationQuestion")

#https://huggingface.co/mrm8488/t5-base-finetuned-question-generation-ap    
model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

from loggerLog import *


def getQuestion(tokenizer, model, answer, context, max_length=64):
    input_text = "answer: %s  context: %s </s>" % (answer, context)
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], 
               attention_mask=features['attention_mask'],
               max_length=max_length)
    return tokenizer.decode(output[0])


def generateQuestionPrependTransformer1(text, answer):
    nlpQuestion = QGPipelines.pipeline("question-generationQuestion")
    result = nlpQuestion([text], [[answer]])

    question = ""
    if result: 
        question = result[0]['question']
    return question


def generateQuestionPrependTransformer2(text, answer):
    nlp = QGPipelines.pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")
    result = nlp(answer+" [SEP] "+text)

    question = ""
    if result: 
        question = result[0]['question']
    return question


def generateQuestionPrependTransformer3(text, answer):
    nlp = QGPipelines.pipeline("question-generation", model="valhalla/t5-small-qg-prepend", qg_format="prepend")
    result = nlp("answer: " + answer + " context: " + text)

    question = ""
    if result: 
        question = result[0]['question']
    return question


def generateQuestionHighlightTransformer(text, answer):
    nlp = QGPipelines.pipeline("question-generation", model="valhalla/t5-base-qg-hl")
    textHighligth = text.replace("answer", "<hl> "+answer+" <hl>", 1) + " </s>"
    result = nlp(textHighligth)

    question = ""
    if result: 
        question = result[0]['question']
    return question


def generateQuestionPrototypeTransformer(tokenizer, model, text, answer):
    question = getQuestion(tokenizer, model, answer, text, max_length=64)[16:-4]
    question = question.replace("<unk>", "")
    return question


def answerAgnosticTransformer(sentence):
    nlp = QGPipelines.pipeline("e2e-qg")
    questions = nlp(sentence)
    return questions[0]


def getQuestions(input, option):
    for i in range(len(input)):
        sentence = input[i]["sentence"]
        for j in range(len(input[i]["answers"])):
            answer = input[i]["answers"][j]["answer"]
            answerLabel = input[i]["answers"][j]["label"]
            input[i]["answers"][j]["questions"] = []
            if option == 1:
                try:
                    question = generateQuestionPrependTransformer1(sentence, answer)
                except Exception as e:
                    logger.error("Exception in function generateQuestionPrependTransformer1 in getQuestions (generatorTransformer.py): %s", e)
                if question != "":
                    input[i]["answers"][j]["questions"].append(question)
            elif option == 2:
                try:
                    question = generateQuestionPrototypeTransformer(tokenizer, model, sentence, answer)
                except Exception as e:
                    logger.error("Exception in function generateQuestionPrototypeTransformer in getQuestions (generatorTransformer.py): %s", e)
                input[i]["answers"][j]["questions"].append(question)
            elif option == 3:
                try:
                    question = answerAgnosticTransformer(sentence)
                except Exception as e:
                    logger.error("Exception in function answerAgnosticTransformer in getQuestions (generatorTransformer.py): %s", e)
                input[i]["answers"][j]["questions"].append(question)
    return input