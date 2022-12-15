import spacy
nlp = spacy.load('en_core_web_lg')
import nltk
import string
from nltk.corpus import stopwords
from nltk.corpus import wordnet 
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from loggerLog import *

def getDistractors(answer, answerLabel, text, penalty, nDistractors):
    entireText = " ".join(text)
    textDoc = nlp(entireText)
    
    distractors = []
    auxDistractors = []

    answers = []
    answerLabels = []
    allTokensAnswers = []
    
    # verify if answer has named entity label
    if answerLabel != "":
        answers = [answer]
        answerLabels = [answerLabel]
    # if not, search for named entities (can be multiple as we consider answer's tokens)
    else:
        try:
            answers, answerLabels = getNamedEntities(answer)
        except Exception as e:
            logger.error("Exception in function getNamedEntities in getDistractors (distractorsNER.py): %s", e)

    # for each named entity present in the answer, search similar named entities with the same label
    for i in range(len(answers)):
        answer = answers[i]
        answerLabel = answerLabels[i]

        answerNLP = nlp(answer)
        try:
            processedAnswer, tokensAnswer = preprocess(answer)
        except Exception as e:
            logger.error("Exception in function preprocess in getDistractors (distractorsNER.py): %s", e)
        allTokensAnswers += tokensAnswer
        try:
            synonymsAnswer = synonymsPhrase(tokensAnswer)
        except Exception as e:
            logger.error("Exception in function synonymsPhrase in getDistractors (distractorsNER.py): %s", e)

        tokensAnswerStem = [ps.stem(tokenAnswer) for tokenAnswer in tokensAnswer]
        for ent in textDoc.ents:
            try:
                processedDistractor, tokensDistractor = preprocess(ent.text)
            except Exception as e:
                logger.error("Exception in function preprocess in getDistractors (distractorsNER.py): %s", e)
            tokensDistractorStem = [ps.stem(tokenDistractor) for tokenDistractor in tokensDistractor]
            # if an entity is different from the answer but has the same label, add to distractors
            if processedDistractor != processedAnswer and ent.label_ == answerLabel and processedDistractor not in auxDistractors:
                auxDistractors.append(processedDistractor)
                # if we can verify spaCy similarity, atribute score similarity minus a penalty if there are synonyms or stems in common
                if answerNLP and answerNLP.vector_norm and ent.vector_norm:
                    similarity = answerNLP.similarity(ent)
                    if (set(synonymsAnswer) & set(synonymsPhrase(tokensDistractor))) or (set(tokensAnswerStem) & set(tokensDistractorStem)):
                        similarity -= abs(similarity)*penalty
                    distractors.append([ent.text, similarity])
                # if we can't verify spaCy similarity attribute score zero
                else:
                    distractors.append([ent.text, 0.0])
        
    # if number of distractors selected is not sufficient, repeat for answer's tokens
    if len(distractors) < nDistractors:
        for token in allTokensAnswers:
            try:
                neTokens, neLabels = getNamedEntities(token)
            except Exception as e:
                logger.error("Exception in function getNamedEntities in getDistractors (distractorsNER.py): %s", e)

            for i in range(len(neTokens)):
                answer = neTokens[i]
                answerLabel = neLabels[i]

                answerNLP = nlp(answer)
                try:
                    synonymsAnswer = synonymsWordnet(answer)
                except Exception as e:
                    logger.error("Exception in function synonymsWordnet in getDistractors (distractorsNER.py): %s", e)

                for ent in textDoc.ents:
                    try:
                        processedDistractor, tokensDistractor =  preprocess(ent.text)
                    except Exception as e:
                        logger.error("Exception in function preprocess in getDistractors (distractorsNER.py): %s", e)
                    if processedDistractor != answer and ent.label_ == answerLabel and processedDistractor not in auxDistractors:
                        auxDistractors.append(processedDistractor)
                        if answerNLP and answerNLP.vector_norm and ent.vector_norm:
                            similarity = answerNLP.similarity(ent)
                            if (set(synonymsAnswer) & set(synonymsPhrase(tokensDistractor))) or (set(tokensAnswer) & set(tokensDistractor)):
                                similarity -= abs(similarity)*penalty
                            distractors.append([ent.text, similarity])
                        else:
                            distractors.append([ent.text, 0.0])
    
    # sort in reverse, as higher scores represent distractors more similar to the answer
    distractors.sort(key = lambda distractors: distractors[1], reverse=True)

    # keep only N best ranked
    finalDistractors = []
    if len(distractors) < nDistractors:
        nDistractors = len(distractors)
    for distractor in distractors[:nDistractors]:
        finalDistractors.append(distractor[0])
    
    return finalDistractors
    

# lower case, remove punctuation and stopwords
def preprocess(distractor):
    distractor = distractor.lower()
    distractor = distractor.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(distractor)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    distractor = " ".join(tokens)
    return distractor, tokens


# search for synonyms using wordnet. synset names use underscores instead of blank spaces
def synonymsWordnet(word):
    word = word.replace(" ","_")
    synonyms = []

    for syn in wordnet.synsets(word):
        for item in syn.lemmas():
            synonym = item.name()
            synonym = synonym.replace("_"," ")
            if synonym not in synonyms:
                synonyms.append(synonym)

    return synonyms


def synonymsPhrase(tokens):
    synonyms = []

    for token in tokens:
        try:
            synonymsAux = synonymsWordnet(token)
        except Exception as e:
            logger.error("Exception in function synonymsWordnet in synonymsPhrase (distractorsNER.py): %s", e)
        for synonym in synonymsAux:
            if synonym not in synonyms:
                synonyms.append(synonym)
    
    return synonyms


def getNamedEntities(text):
    doc = nlp(text)
    neTokens = []
    neLabels = []
    if doc.ents:
        for ent in doc.ents:
            neTokens.append(ent.text)
            neLabels.append(ent.label_)
    return neTokens, neLabels
