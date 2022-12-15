from nltk.corpus import wordnet
import string
import nltk
from nltk.corpus import stopwords
import spacy
nlp = spacy.load('en_core_web_lg')
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from loggerLog import *

def getDistractors(answer, penalty, nDistractors):
    distractors = []
    try:
        phrase, tokens = preprocessPhrase(answer)
    except Exception as e:
        logger.error("Exception in function preprocessPhrase in getDistractors (distractorsWordNet.py): %s", e)

    # we try to get potential distractors looking for cohyponyms
    # i.e. words that share the same hyperonym
    try:
        cohyponyms = getCohyponyms(answer)
    except Exception as e:
        logger.error("Exception in function getCohyponyms in getDistractors (distractorsWordNet.py): %s", e)
    for cohyponym in cohyponyms:
        if cohyponym != answer and cohyponym != phrase and cohyponym not in distractors:
            distractors.append(cohyponym)
    
    # if number of distractors selected is not sufficient, repeat for processed answer
    if len(distractors) < nDistractors:
        try:
            cohyponyms = getCohyponyms(phrase)
        except Exception as e:
            logger.error("Exception in function getCohyponyms in getDistractors (distractorsWordNet.py): %s", e)
        for cohyponym in cohyponyms:
            if cohyponym != answer and cohyponym != phrase and cohyponym not in distractors:
                distractors.append(cohyponym)

    # if number of distractors selected is still not sufficient, repeat for answer's tokens
    if len(distractors) < nDistractors:
        for token in tokens:
            try:
                cohyponymsToken = getCohyponyms(token)
            except Exception as e:
                logger.error("Exception in function getCohyponyms in getDistractors (distractorsWordNet.py): %s", e) 
            for cohyponymToken in cohyponymsToken:
                if cohyponymToken != answer and cohyponymToken != phrase and cohyponymToken not in distractors:
                    distractors.append(cohyponymToken)
    
    answerNLP = nlp(answer)
    tokensAnswerStem = [ps.stem(tokenAnswer) for tokenAnswer in tokens]
    for i in range(len(distractors)):
        distractorNLP = nlp(distractors[i])
        try:
            phraseDistractor, tokensDistractor = preprocessPhrase(distractors[i])
        except Exception as e:
            logger.error("Exception in function preprocessPhrase in getDistractors (distractorsWordNet.py): %s", e)
        tokensDistractorStem = [ps.stem(tokenDistractor) for tokenDistractor in tokensDistractor]
        # if we can verify spaCy similarity, atribute score similarity minus a penalty if there are tokens in common
        if answerNLP.vector_norm and distractorNLP.vector_norm:
            similarity = answerNLP.similarity(distractorNLP)      
            if (set(tokensAnswerStem) & set(tokensDistractorStem)):
                similarity -= abs(similarity)*penalty
            distractors[i] = [distractors[i], similarity]
        # if we can't verify spaCy similarity attribute score zero
        else:
            distractors[i] = [distractors[i], 0.0]

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
def preprocessPhrase(phrase):
    phrase = phrase.lower()
    phrase = phrase.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(phrase)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    return phrase, tokens


# cohyponyms share the same hyperonym
# first we look for hyperonyms of the word, then proceed to get the hyponyms of those hyperonyms
def getCohyponyms(word):
    word = word.replace(" ","_")

    cohyponyms = []
    synonyms = wordnet.synsets(word)
    
    for synonym in synonyms:
        hypernyms = synonym.hypernyms()
        for hypernym in hypernyms:
            hyponyms = hypernym.hyponyms()
            for hyponym in hyponyms:
                if hyponym not in synonyms:
                    lemmas = hyponym.lemmas()
                    for lemma in lemmas:
                        name = lemma.name()
                        name = name.replace("_"," ")
                        cohyponyms.append(name)
    
    return cohyponyms 

