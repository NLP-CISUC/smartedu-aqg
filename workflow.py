import answerSelection
import generatorRules
import generatorTransformer
import distractorsNER
import distractorsWordNet
import distractorsDBpedia
import distractorsGlove
import distractorsTransformer
import pdfText
from corefResol import spacyExperimentalCoref
from questionRanking import summarizeTF_IDF
from distractorRanking import sortDistractorsGPT2
from questionRanking import scoreGPT2
from loggerLog import *


def allSteps(text, answerOption, questionOption, distractorOption, maxNumberQuestions):
    completeOutput = []

    # COREFERENCE RESOLUTION
    try:
        logger.info("Performing coreference resolution")
        text = spacyExperimentalCoref(text)
    except Exception as e:
        logger.error("Exception while performing coreference resolution (spacyExperimentalCoref): %s", e)

    # SUMMARY
    try:
        logger.info("Performing text summarization")
        if maxNumberQuestions == "default":
            text = summarizeTF_IDF(text, 50)
        else:
            text = summarizeTF_IDF(text, int(maxNumberQuestions))
    except Exception as e:
        logger.error("Exception while performing text summarization (summarizeTF_IDF): %s", e)
    
    output = []
    # ANSWER SELECTION
    try:
        logger.info("Performing answer selection")
        # 0:named entities, 1:noun chunks, 2:transformer
        if answerOption in ["0","1","2"]:
            output = answerSelection.excerptAnswers(text, int(answerOption))
        # default:named entities
        elif answerOption == "default":
            output = answerSelection.excerptAnswers(text, 0)
    except Exception as e:
        logger.error("Exception while performing answer selection (answerSelection.excerptAnswers): %s", e)

    # QUESTION GENERATION
    # 0: rules, 1:prepend transformer, 2:prototype transformer, 3:e2e
    try:
        logger.info("Performing question generation")
        if questionOption == "0":
            output = generatorRules.getQuestions(output)
        elif questionOption in ["1","2","3"]:
            output = generatorTransformer.getQuestions(output, int(questionOption))
        elif questionOption == "default":
            output = generatorTransformer.getQuestions(output, 2)
    except Exception as e:
        logger.error("Exception while performing question generation: %s", e)

    # before selecting distractors, we want to remain with only one question per sentence
    try:
        output = questionSentence(output)
    except Exception as e:
        logger.error("Exception in function questionSentence in allSteps (workflow.py): %s", e)

    distractorsPenalty = 0.5
    nDistractors = 5
    # DISTRACTOR GENERATION
    try:
        logger.info("Performing distractor selection")
        for i in range(len(output)):
            for j in range(len(output[i]["answers"])):
                questions = output[i]["answers"][j]["questions"]
                if questions:
                    answer = output[i]["answers"][j]["answer"]
                    if distractorOption == "1" or distractorOption == "6":
                        answerLabel = output[i]["answers"][j]["label"]
                        output[i]["answers"][j]["distractorsNER"] = distractorsNER.getDistractors(answer, answerLabel, text, distractorsPenalty, nDistractors)
                    if distractorOption == "2" or distractorOption == "6":
                        output[i]["answers"][j]["distractorsWordNet"] = distractorsWordNet.getDistractors(answer, distractorsPenalty, nDistractors)
                    if distractorOption == "3" or distractorOption == "6":
                        output[i]["answers"][j]["distractorsDBpedia"] = distractorsDBpedia.getDistractors(answer, nDistractors, distractorsPenalty)
                    if distractorOption == "4" or distractorOption == "6":
                        output[i]["answers"][j]["distractorsGlove"] = distractorsGlove.getDistractors(answer, nDistractors, distractorsPenalty)
                    if distractorOption == "5" or distractorOption == "6":
                        question = questions[0]
                        output[i]["answers"][j]["distractorsTransformer"] = distractorsTransformer.getDistractors(text, question, answer, nDistractors)
                    elif distractorOption == "default":  #default: GloVe
                        output[i]["answers"][j]["distractorsGlove"] = distractorsGlove.getDistractors(answer, nDistractors, distractorsPenalty)
    except Exception as e:
        logger.error("Exception while performing distractor selection: %s", e)

    completeOutput += output
    finalOutput = outputAPI(completeOutput)
    logger.info("Returning %s questions", len(finalOutput))
    return finalOutput


def allStepsPDF(filename, answerSelection, questionGeneration, distractorSelection, maxNumberQuestions):
    try:
        logger.info("Reading PDF")
        text = pdfText.readPdf(filename)
    except Exception as e:
        logger.error("Exception in pdfText.readPdf in allStepsPDF (workflow.py): %s", e)
    return allSteps(text, answerSelection, questionGeneration, distractorSelection, maxNumberQuestions)
   

def outputAPI(originalOutput):
    newOutput = []

    for item in originalOutput:
        for answer in item["answers"]:
            dic = {}
            dic["question"] = answer["questions"][0]
            dic["answers"] = {"correct_answers": [], "wrong_answers": []}
            dic["answers"]["correct_answers"] = [answer["answer"]]
            distractorKeys = ["distractorsNER", "distractorsWordNet", "distractorsDBpedia", "distractorsGlove", "distractorsTransformer"]
            for key in distractorKeys:
                if key in answer.keys():
                    dic["answers"]["wrong_answers"] += answer[key]
            # SORT DISTRACTORS
            dic["answers"]["wrong_answers"] = list(sortDistractorsGPT2(dic["question"], dic["answers"]["wrong_answers"]).keys())
            newOutput.append(dic)
    
    return newOutput


# limit to only one question per sentence
def questionSentence(originalOutput):
    newOutput = []

    for item in originalOutput:
        bestAnswerItem = []
        for answerItems in item['answers']:
            if len(answerItems['questions']) > 0:
                answerItems['questions'] = [answerItems['questions'][0]]
                answer = answerItems['answer']
                question = answerItems['questions'][0]
                if(bestAnswerItem == []):
                    bestAnswerItem = [answerItems, scoreGPT2(question+" "+answer)]
                else:
                    score = scoreGPT2(question+" "+answer)
                    if score < bestAnswerItem[1]:
                        bestAnswerItem = [answerItems, score]
                        
        if len(bestAnswerItem) > 0:
            newOutput.append({'sentence':item['sentence'], 'answers': [bestAnswerItem[0]]})
    
    return newOutput

    