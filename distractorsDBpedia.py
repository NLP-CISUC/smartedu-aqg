import spacy
nlp = spacy.load('en_core_web_lg')
from SPARQLWrapper import SPARQLWrapper, JSON
sparql = SPARQLWrapper('https://dbpedia.org/sparql')
import string
import nltk
from nltk.corpus import stopwords
import distractorsNER
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from loggerLog import *

# search for DBpedia URIs that have as label the given string in the english language
def searchURIs(value):
    URIs = []

    query = '''
            SELECT ?uri
            WHERE {
                ?uri rdfs:label "'''+value+'''"@en .
            }
            '''
        
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        qres = sparql.query().convert()
        for binding in qres['results']['bindings']:
            URIs.append(binding['uri']['value'])
    except:
        pass

    return URIs


# search for broader concepts (hyperonyms) of the given URI
def skosBroader(value):
    URIs = []

    query = '''
        SELECT ?value
        WHERE { 
            <'''+value+'''> skos:broader ?value .
        }
        '''
    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        qres = sparql.query().convert()
        for binding in qres['results']['bindings']:
            URIs.append(binding['value']['value'])
    except:
        pass

    return URIs


# search for narrower concepts (hyponyms) of the given URI
def skosNarrower(value):
    labels = []

    query = '''
        SELECT ?value ?label
        WHERE { 
            <'''+value+'''> ^skos:broader ?value .
            ?value rdfs:label ?label .
            filter langMatches(lang(?label),"en")
        }
        '''
    
    try:
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        qres = sparql.query().convert()
        for binding in qres['results']['bindings']:
            labels.append(binding['label']['value'])
    except:
        pass
    
    return labels


""" Search for cohyponyms of the answer. We can caracterize cohyponyms as concepts that have the same hyperonym.
First we search for broader concepts (hyperonym) of the answer and then, for each hyperonym, search for its hyponyms.
"""
def cohyponymsBroaderNarrower(answer):
    distractors = []
    try:
        labels = searchURIs(answer)
    except Exception as e:
        logger.error("Exception in function searchURIs in cohyponymsBroaderNarrower (distractorsDBpedia.py): %s", e)

    if labels:
        for label in labels:
            try:
                broaderConcepts = skosBroader(label)
            except Exception as e:
                logger.error("Exception in function skosBroader in cohyponymsBroaderNarrower (distractorsDBpedia.py): %s", e)
            for broaderConcept in broaderConcepts:
                try:
                    narrowerConcepts = skosNarrower(broaderConcept)
                except Exception as e:
                    logger.error("Exception in function skosNarrower in cohyponymsBroaderNarrower (distractorsDBpedia.py): %s", e)
                for narrowerConcept in narrowerConcepts:
                    if narrowerConcept not in distractors and narrowerConcept != answer:
                        distractors.append(narrowerConcept)

    return distractors


# search for type of the URI
def rdf_type(value):
    query = '''
        SELECT ?value
        WHERE { 
            <'''+value+'''> rdf:type ?value .
        }
        '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()

    URIs = []
    for binding in qres['results']['bindings']:
        uri = binding['value']['value']
        if uri != "http://www.w3.org/2002/07/owl#Thing":
            URIs.append(binding['value']['value'])
    
    return URIs


# search for concepts with given type
def rdf_typeInverse(uri):
    query = '''
        SELECT ?value ?label
        WHERE { 
            <'''+uri+'''> ^rdf:type ?value .
            ?value rdfs:label ?label .
            filter langMatches(lang(?label),"en")
        }
        '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    
    labels = []
    for binding in qres['results']['bindings']:
        labels.append(binding['label']['value'])

    return labels


""" Search for related concepts, only in this case we try to translate the 
logic of cohyponyms to concepts that share the same type. """
def cohyponymsType(answer):
    distractors = []
    try:
        labels = searchURIs(answer)
    except Exception as e:
        logger.error("Exception in function searchURIs in cohyponymsType (distractorsDBpedia.py): %s", e)

    if labels:
        for label in labels:
            try:
                types = rdf_type(label)
            except Exception as e:
                logger.error("Exception in function rdf_type in cohyponymsType (distractorsDBpedia.py): %s", e)
            for type in types:
                try:
                    coHyponyms = rdf_typeInverse(type)
                except Exception as e:
                    logger.error("Exception in function rdf_typeInverse in cohyponymsType (distractorsDBpedia.py): %s", e)
                for coHyponym in coHyponyms:
                    if coHyponym not in distractors:
                        distractors.append(coHyponym)
    
    return distractors


# search for subclass of the URI
def rdfs_subclassOf(uri):
    query = '''
        SELECT ?value
        WHERE { 
            <'''+uri+'''> rdfs:subclassOf ?value .
        }
        '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    URIs = []
    for binding in qres['results']['bindings']:
        URIs.append(binding['value']['value'])
    
    return URIs


# search for concepts with same subclass
def rdfs_subclassOfInverse(uri):
    query = '''
        SELECT ?value ?label
        WHERE { 
            <'''+uri+'''> ^rdfs:subclassOf ?value .
            ?value rdfs:label ?label .
        }
        '''
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    qres = sparql.query().convert()
    
    labels = []
    for binding in qres['results']['bindings']:
        labels.append(binding['label']['value'])
    
    return qres


""" Search for related concepts, only in this case we try to translate the 
logic of cohyponyms to concepts that share the same subclass. """
def cohyponymsSubclass(answer):
    distractors = []
    try:
        labels = searchURIs(answer)
    except Exception as e:
        logger.error("Exception in function searchURIs in cohyponymsSubclass (distractorsDBpedia.py): %s", e)

    if labels:
        for label in labels:
            try:
                hypernyms = rdfs_subclassOf(label)
            except Exception as e:
                logger.error("Exception in function rdfs_subclassOf in cohyponymsSubclass (distractorsDBpedia.py): %s", e)
            for hypernym in hypernyms:
                try:
                    coHyponyms = rdfs_subclassOfInverse(hypernym)
                except Exception as e:
                    logger.error("Exception in function rdfs_subclassOfInverse in cohyponymsSubclass (distractorsDBpedia.py): %s", e)
                for coHyponym in coHyponyms:
                    if coHyponym not in distractors:
                        distractors.append(coHyponym)
    
    return distractors


# attribute scores to distractors, sort them and keep only N best scores
def filterDistractors(answer, tokensAnswer, distractors, nDistractors, penalty):
    finalDistractors = []
    answerNLP = nlp(answer)
    # get synonyms of the answer (or answer's tokens) to verify the presence of synonyms in the distractors
    try:
        synonymsAnswer = distractorsNER.synonymsPhrase(tokensAnswer)
    except Exception as e:
        logger.error("Exception in function distractorsNER.synonymsPhrase in filterDistractors (distractorsDBpedia.py): %s", e)

    # get answer tokens' stems to verify if it as stems in common with distractors
    tokensAnswerStem = [ps.stem(tokenAnswer) for tokenAnswer in tokensAnswer]

    for i in range(len(distractors)):
        distractorNLP = nlp(distractors[i])
        try:
            processedDistractor, tokensDistractor = preprocessDistractor(distractors[i])
        except Exception as e:
            logger.error("Exception in function preprocessDistractor in filterDistractors (distractorsDBpedia.py): %s", e)
        # get distractors tokens' stems to verify if they have stems in common with the answer
        tokensDistractorStem = [ps.stem(tokenDistractor) for tokenDistractor in tokensDistractor]
        # if we can verify spaCy similarity, atribute score similarity minus a penalty if there are synonyms or stems in common
        if answerNLP.vector_norm and distractorNLP.vector_norm:
            similarity = answerNLP.similarity(distractorNLP)    
            if (set(synonymsAnswer) & set(distractorsNER.synonymsPhrase(tokensDistractor))) or (set(tokensAnswerStem) & set(tokensDistractorStem)):        
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
    

# main function 
def getDistractors(answer, nDistractors, penalty):
    try:
        processedAnswer, tokensAnswer = preprocessDistractor(answer)
    except Exception as e:
        logger.error("Exception in function preprocessDistractor in getDistractors (distractorsDBpedia.py): %s", e)
    
    # first letter upper case and lower case to maximize number of distractors discovered
    if len(answer)>1:
        possibleAnswers = [answer[0].lower() + answer[1:], answer[0].upper() + answer[1:]]
    else:
        possibleAnswers = [answer]

    try:
        distractors = selectDistractors(possibleAnswers, nDistractors)
    except Exception as e:
        logger.error("Exception in function selectDistractors in getDistractors (distractorsDBpedia.py): %s", e)

    # if number of distractors selected is not sufficient, repeat for processed answer
    if len(distractors) < nDistractors and len(processedAnswer) > 1:
        possibleAnswers = [processedAnswer[0].lower() + processedAnswer[1:], processedAnswer[0].upper() + processedAnswer[1:]]
        try:
            newDistractors = selectDistractors(possibleAnswers, nDistractors)
        except Exception as e:
            logger.error("Exception in function selectDistractors in getDistractors (distractorsDBpedia.py): %s", e)
        for distractor in newDistractors:
            if distractor not in distractors:
                distractors.append(distractor)
    
    # if number of distractors selected still not sufficient, repeat answer's tokens
    if len(distractors) < nDistractors:
        possibleAnswers = []
        for tokenAnswer in tokensAnswer:
            if len(tokenAnswer) > 1:
                possibleAnswers.append(tokenAnswer[0].lower() + tokenAnswer[1:])
                possibleAnswers.append(tokenAnswer[0].upper() + tokenAnswer[1:])
        try:
            newDistractors = selectDistractors(possibleAnswers, nDistractors)
        except Exception as e:
            logger.error("Exception in function selectDistractors in getDistractors (distractorsDBpedia.py): %s", e)
        for distractor in newDistractors:
            if distractor not in distractors:
                distractors.append(distractor)


    # unique, similar and sorted distractors
    try:
        distractors = filterDistractors(answer, tokensAnswer, distractors, nDistractors, penalty)
    except Exception as e:
        logger.error("Exception in function filterDistractors in getDistractors (distractorsDBpedia.py): %s", e)

    return distractors


# cohyponymsBroaderNarrower was the only function that seemed to have interesting results
def selectDistractors(possibleAnswers, nDistractors):
    distractors = []
    count = 0

    for possibleAnswer in possibleAnswers:
        try:
            distractorsAux = cohyponymsBroaderNarrower(possibleAnswer)
        except Exception as e:
            logger.error("Exception in function cohyponymsBroaderNarrower in selectDistractors (distractorsDBpedia.py): %s", e)
        for distractorAux in distractorsAux:
            if distractorAux not in distractors:
                distractors.append(distractorAux)
                count += 1

    """
    if count < nDistractors*5:
        for possibleAnswer in possibleAnswers:
            try:
                distractorsAux = cohyponymsType(possibleAnswer)
            except Exception as e:
                logger.error("Exception in function cohyponymsType in selectDistractors (distractorsDBpedia.py): %s", e)
            for distractorAux in distractorsAux:
                if distractorAux not in distractors:
                    distractors.append(distractorAux)
    
    if count < nDistractors*5:
        for possibleAnswer in possibleAnswers:
            try:
                distractorsAux = cohyponymsSubclass(possibleAnswer)
            except Exception as e:
                logger.error("Exception in function cohyponymsSubclass in selectDistractors (distractorsDBpedia.py): %s", e)
            for distractorAux in distractorsAux:
                if distractorAux not in distractors:
                    distractors.append(distractorAux)
    """

    return distractors


# lower case, remove punctuation and stopwords
def preprocessDistractor(distractor):
    distractor = distractor.lower()
    distractor = distractor.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(distractor)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    distractor = " ".join(tokens)
    return distractor, tokens

