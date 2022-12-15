import spacy
nlp = spacy.load('en_core_web_lg')
from nltk.corpus import stopwords
from nltk.corpus import wordnet 
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import gensim.downloader
#print(list(gensim.downloader.info()['models'].keys()))     #all available models in gensim-data
glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100') #wiki embeddings
import distractorsNER
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from loggerLog import *

# returns distractors and their scores
def gloveDistractors(answer, nDistractors):
    distractors = []
    similars = []

    # this try is not for logging
    try:
        similars = glove_vectors.most_similar(answer, topn=nDistractors)
    except:
        pass

    for similar in similars:
        similarList = [similar[0], similar[1]]
        distractors.append(similarList)

    return distractors


# main function
def getDistractors(answer, nDistractors, penalty):
    answerNLP = nlp(answer)
    try:
        processedAnswer, tokensAnswer =  distractorsNER.preprocess(answer)
    except Exception as e:
        logger.error("Exception in function distractorsNER.preprocess in getDistractors (distractorsGlove.py): %s", e)
    try:
        synonymsAnswer = distractorsNER.synonymsPhrase(tokensAnswer)
    except Exception as e:
        logger.error("Exception in function distractorsNER.synonymsPhrase in getDistractors (distractorsGlove.py): %s", e)
    try:
        distractors = gloveDistractors(answer, nDistractors)
    except Exception as e:
        logger.error("Exception in function gloveDistractors in getDistractors (distractorsGlove.py): %s", e)

    # if number of distractors selected is not sufficient, try the same for processed answer
    if len(distractors) < nDistractors:
        try:
            newDistractors = gloveDistractors(processedAnswer, nDistractors)
        except Exception as e:
            logger.error("Exception in function gloveDistractors in getDistractors (distractorsGlove.py): %s", e)
        onlyDistractors = [item[0] for item in distractors]
        for newDistractor in newDistractors:
            if newDistractor[0] not in onlyDistractors:
                distractors.append(newDistractor)

    # if number of distractors selected still not sufficient, try the same for answer's tokens
    if len(distractors) < nDistractors:
        for tokenAnswer in tokensAnswer:
            try:
                distractorsToken = gloveDistractors(tokenAnswer, nDistractors)
            except Exception as e:
                logger.error("Exception in function gloveDistractors in getDistractors (distractorsGlove.py): %s", e)
            onlyDistractors = [item[0] for item in distractors]
            for distractorToken in distractorsToken:
                distractorToken[0] = answer.lower().replace(tokenAnswer, distractorToken[0])
                if distractorToken[0] not in onlyDistractors:
                    distractors.append(distractorToken)
    
    # if answer and distractors have synonyms or tokens in common, penalyze their scores
    tokensAnswerStem = [ps.stem(tokenAnswer) for tokenAnswer in tokensAnswer]
    for i in range(len(distractors)):
        distractorNLP = nlp(distractors[i][0])
        try:
            processedDistractor, tokensDistractor =  distractorsNER.preprocess(distractors[i][0])
        except Exception as e:
            logger.error("Exception in function distractorsNER.preprocess in getDistractors (distractorsGlove.py): %s", e)
        tokensDistractorStem = [ps.stem(tokenDistractor) for tokenDistractor in tokensDistractor]
        if answerNLP.vector_norm and distractorNLP.vector_norm:
            if (set(synonymsAnswer) & set(distractorsNER.synonymsWordnet(distractors[i][0]))) or (set(tokensAnswerStem) & set(tokensDistractorStem)):
                distractors[i][1] -= abs(distractors[i][1])*penalty

    # sort in reverse, as higher scores represent distractors more similar to the answer
    distractors.sort(key = lambda distractors: distractors[1], reverse=True)

    # keep only N best ranked
    finalDistractors = [item[0] for item in distractors]
    if len(distractors) < nDistractors:
        nDistractors = len(distractors)
    finalDistractors = finalDistractors[:nDistractors]

    return finalDistractors
    
