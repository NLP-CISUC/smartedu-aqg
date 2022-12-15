import spacy
nlp = spacy.load('en_core_web_lg')
import nltk
from nltk.corpus import stopwords
stopwords = stopwords.words('english')
import question_generation.pipelines as QGPipelines #adapted from https://github.com/patil-suraj/question_generation
from loggerLog import *

""" Main method to get potencial answers from a text excerpt.
we also try to obtain the named entity label of the answer (or, if not possible, of one of its tokens).
Options: 0 -> named entities, 1 -> noun chunks, 2 -> transformer """
def excerptAnswers(excerpt, option):
    try:
        sentences = selectSentences(excerpt)
    except Exception as e:
        logger.error("Exception in function selectSentences in excerptAnswers (answerSelection.py): %s", e)
    outputAnswerSelection = []

    if option == 2:
        # load pipeline ("question-generationAnswer" was adapted from "question-generation" to only perform answer selection)
        try:
            nlp = QGPipelines.pipeline("question-generationAnswer")
        except Exception as e:
            logger.error("Exception loading model question-generationAnswer in excerptAnswers (answerSelection.py): %s", e)

    for sentence in sentences:
        if option == 0 or option == 1:
            try:
                answers, answersLabels = answersAndLabels(sentence, option)
            except Exception as e:
                logger.error("Exception in function answersAndLabels in excerptAnswers (answerSelection.py): %s", e)
        elif option == 2:
            try:
                answers, answersLabels = answersTransformer(sentence, nlp)
            except Exception as e:
                logger.error("Exception in function answersTransformer in excerptAnswers (answerSelection.py): %s", e)
            
        if answers:
            sentenceDict = {}
            sentenceDict["sentence"] = sentence

            answersList = []
            for i in range(len(answers)):
                answerDict = {"answer": answers[i], "label": answersLabels[i]}
                answersList.append(answerDict)
            
            sentenceDict["answers"] = answersList
            outputAnswerSelection.append(sentenceDict)

    return outputAnswerSelection


""" Returns a list of the sentences. Removes whitespaces from the beginning and end of each sentence, and eliminates
possible strings only constituted by whitespaces """
def selectSentences(excerpt):
    selected = []
    doc = nlp(excerpt)
    for sentence in doc.sents:
        sentenceString = str(sentence)
        # verifies if sentence still exists after removing whitespaces from the beginning and end
        if sentenceString.strip():
            selected.append(sentenceString)
    return selected


""" used to get named entities (NEs, method 0) or noun chunks (NCs, method 1) as potential answers.
In future steps of the workflow (e.g. distractor selection), it is useful to to have answers' NE label answers obtained
by analizing NEs already have a corresponding label, we also try to correspond a label to the ones obtained from NCs. """
def answersAndLabels(sentence, method):
    answers = []
    labels = []
    if method == 0:
        # answers and their labels according to named entity recognition
        try:
            neTokens, neLabels = getNamedEntities(sentence)
        except Exception as e:
            logger.error("Exception in function getNamedEntities in answersAndLabels (answerSelection.py): %s", e)
        answers = neTokens
        labels = neLabels
    elif method == 1:
        # noun chunks as answers
        try:
            nounChunks = getNounChunks(sentence)
        except Exception as e:
            logger.error("Exception in function getNounChunks in answersAndLabels (answerSelection.py): %s", e)
        ncLabels = []
        # we still get the answers according to NER to then compare with NCs
        try:
            neTokens, neLabels = getNamedEntities(sentence)
        except Exception as e:
            logger.error("Exception in function getNamedEntities in answersAndLabels (answerSelection.py): %s", e)
        for nounChunk in nounChunks:
            # if a noun chunks is equal to a named entity, we simply attribute the same label
            if nounChunk in neTokens:
                index = neTokens.index(nounChunk)
                ncLabels.append(neLabels[index])
            else:
                # if not, after processing the noun chunk, we repeat the process to try getting a label that corresponds to one of its tokens
                try:
                    nounChunk = processPhrase(nounChunk)
                except Exception as e:
                    logger.error("Exception in function processPhrase in answersAndLabels (answerSelection.py): %s", e)
                try:
                    innerNETokens, innerNELabels = getNamedEntities(nounChunk)
                except Exception as e:
                    logger.error("Exception in function getNamedEntities in answersAndLabels (answerSelection.py): %s", e)
                # if multiple labels are found, we simply attribute the first to the whole noun chunk
                if innerNELabels:
                    ncLabels.append(innerNELabels[0])  
                # if there is still no label found, we attribute an empty string  
                else:
                    ncLabels.append("")
        answers = nounChunks
        labels = ncLabels
    return answers, labels


# returns named entities present in a string and their labels
def getNamedEntities(text):
    doc = nlp(text)
    neTokens = []
    neLabels = []
    if doc.ents:
        for ent in doc.ents:
            neTokens.append(ent.text)
            neLabels.append(ent.label_)
    return neTokens, neLabels


# returns noun chunks present in a string
def getNounChunks(text):
    doc = nlp(text)
    chunks = []
    if doc.noun_chunks:
        for chunk in doc.noun_chunks:
            chunks.append(chunk.text)
    return chunks


# removes punctuation and stopwords from a string
def processPhrase(phrase):
    # same as string.punctuation but without the apostrophe
    punctuation = "!\"#$%&()*+,-./:;<=>?@[\]^_`{|}~"
    phrase = phrase.translate(str.maketrans('', '', punctuation))
    tokens = nltk.word_tokenize(phrase)
    tokens = [t for t in tokens if t not in stopwords]
    phrase = " ".join(tokens)
    return phrase


""" Uses the transformer (directory question_generation) to obtain potential answers (method 2).
The process to attribute a named entity label to a answer is equal to the what is performed for noun chunks. """
def answersTransformer(text, nlp):
    # potential answers obtained using the pipeline "question-generationAnswer"
    answers = nlp(text)
    
    labels = []
    # we still get the answers according to NER to then compare with the ones obtained with the transformer
    try:
        neTokens, neLabels = getNamedEntities(text)
    except Exception as e:
        logger.error("Exception in function getNamedEntities in answersTransformer (answerSelection.py): %s", e)    
    for answer in answers:
        # if an answer is equal to a named entity, we simply attribute the same label
        if answer in neTokens:
            index = neTokens.index(answer)
            labels.append(neLabels[index])
        else:
            # if not, after processing the answer, we repeat the process to try getting a label that corresponds to one of its tokens
            try:
                answer = processPhrase(answer)
            except Exception as e:
                logger.error("Exception in function processPhrase in answersTransformer (answerSelection.py): %s", e)    
            try:
                innerNETokens, innerNELabels = getNamedEntities(answer)
            except Exception as e:
                logger.error("Exception in function getNamedEntities in answersTransformer (answerSelection.py): %s", e)    
            # if multiple labels are found, we simply attribute the first to the whole noun chunk
            if innerNELabels:
                labels.append(innerNELabels[0])
            # if there is still no label found, we attribute an empty string    
            else:
                labels.append("")
    return answers, labels
