import spacy
import claucy
nlp = spacy.load('en_core_web_lg')
claucy.add_to_pipe(nlp)
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
from loggerLog import *


""" Clauses can be of type SV, SVA, SVO, SVOA, SVC, SVOC or SVOO.
These types are dertermined by the components of the clause: subject (S),
verb (V), adverbial (A), direct object (O), indirect object (O),
complement (C). """
def produceQuestion(sentence, answer, label):
    QApairs = []
    doc = nlp(sentence)
    for sentence in doc.sents:
        for clause in sentence._.clauses:
            verbTag = sentence.root.tag_
            if str(clause.verb) != "None":
                if clause.type == "SVOO":
                    try:
                        QApairs += svoo(verbTag, clause, answer, label)
                    except Exception as e:
                        logger.error("Exception in function svoo in produceQuestion (generatorRules.py): %s", e)
                elif clause.type == "SVOC":
                    try:
                        QApairs += svoc(verbTag, clause, answer, label)
                    except Exception as e:
                        logger.error("Exception in function svoc in produceQuestion (generatorRules.py): %s", e)
                elif clause.type in ["SVO","SVOA"]:
                    try:
                        QApairs += svo(verbTag, clause, answer, label)
                    except Exception as e:
                        logger.error("Exception in function svo in produceQuestion (generatorRules.py): %s", e)
                elif clause.type == "SVC":
                    try:
                        QApairs += svc(verbTag, clause, answer, label)
                    except Exception as e:
                        logger.error("Exception in function svc in produceQuestion (generatorRules.py): %s", e)
                elif clause.type in ["SV","SVA"]:
                    try:
                        QApairs += sv(verbTag, clause, answer, label)
                    except Exception as e:
                        logger.error("Exception in function sv in produceQuestion (generatorRules.py): %s", e)
    return QApairs


def sv(verbTag, clause, answer, label):
    QApairs = []
    subject = str(clause.subject)
    if answer in subject:
        try:
            QApairs = sv_subject(clause, answer, label)
        except Exception as e:
            logger.error("Exception in function sv_subject in sv (generatorRules.py): %s", e)
    else:
        for j in range(len(clause.adverbials)):
            adverbial = str(clause.adverbials[j])
            if answer in adverbial:
                try:
                    QApairs = sv_adverbial(verbTag, clause, answer, label)
                except Exception as e:
                    logger.error("Exception in function sv_adverbial in sv (generatorRules.py): %s", e)
    return QApairs


def sv_subject(clause, answer, label):
    QApairs = []
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in sv_subject (generatorRules.py): %s", e)

    question = " ".join([pronoun, str(clause.verb), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    
    return QApairs


def sv_adverbial(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in sv_adverbial (generatorRules.py): %s", e)
    #pronoun = "How"
    pronoun = questionPronoun(label)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1]]) + "?"
    else:
        aux = auxVerb(verbTag, label)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1]]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])      
    return QApairs


def svo(verbTag, clause, answer, label):
    QApairs = []
    subject = str(clause.subject)
    directObject = str(clause.direct_object)
    if answer in subject:
        try:
            QApairs = svo_subject(clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svo_subject in svo (generatorRules.py): %s", e)
    elif answer in directObject:
        try:
            QApairs = svo_dirObject(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svo_dirObject in svo (generatorRules.py): %s", e)
    else:
        for j in range(len(clause.adverbials)):
            adverbial = str(clause.adverbials[j])
            if answer in adverbial:
                try:
                    QApairs = svo_adverbial(verbTag, clause, answer, label)
                except Exception as e:
                    logger.error("Exception in function svo_adverbial in svo (generatorRules.py): %s", e)
    return QApairs


def svo_subject(clause, answer, label):
    QApairs = []
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svo_subject (generatorRules.py): %s", e)
    question = " ".join([pronoun, str(clause.verb), str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svo_dirObject(verbTag, clause, answer, label): 
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svo_dirObject (generatorRules.py): %s", e)
    try:
        pronoun = questionPronoun(label)   
    except Exception as e:
        logger.error("Exception in function questionPronoun in svo_dirObject (generatorRules.py): %s", e)  
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], " ".join(str(string) for string in clause.adverbials)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svo_dirObject (generatorRules.py): %s", e)  
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svo_adverbial(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svo_adverbial (generatorRules.py): %s", e)  
    #pronoun = "How"
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svo_adverbial (generatorRules.py): %s", e)  

    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.direct_object)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.direct_object)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svo_adverbial (generatorRules.py): %s", e)  
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.direct_object)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])            
    return QApairs


def svc(verbTag, clause, answer, label):
    QApairs = []
    subject = str(clause.subject)
    complement = str(clause.complement)
    if answer in subject:
        try:
            QApairs = svc_subject(clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svc_subject in svc (generatorRules.py): %s", e)  
    elif answer in complement:
        try:
            QApairs = svc_complement(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svc_complement in svc (generatorRules.py): %s", e)  
    else:
        for j in range(len(clause.adverbials)):
            adverbial = str(clause.adverbials[j])
            if answer in adverbial:
                try:
                    QApairs = svc_adverbial(verbTag, clause, answer, label)
                except Exception as e:
                    logger.error("Exception in function svc_adverbial in svc (generatorRules.py): %s", e)  
    return QApairs


def svc_subject(clause, answer, label):
    QApairs = []
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svc_subject (generatorRules.py): %s", e)  
    question = " ".join([pronoun, str(clause.verb), str(clause.complement), " ".join(str(string) for string in clause.adverbials)]) + "?"     
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svc_complement(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svc_complement (generatorRules.py): %s", e)  
    try:
        pronoun = questionPronoun(label)   
    except Exception as e:
        logger.error("Exception in function questionPronoun in svc_complement (generatorRules.py): %s", e)   
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], " ".join(str(string) for string in clause.adverbials)]) + "?"         
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svc_complement (generatorRules.py): %s", e)   
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])  
    return QApairs


def svc_adverbial(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svc_adverbial (generatorRules.py): %s", e)   
    #pronoun = "How"
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svc_adverbial (generatorRules.py): %s", e)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.complement)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.complement)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svc_adverbial (generatorRules.py): %s", e)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.complement)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])        
    return QApairs


def svoc(verbTag, clause, answer, label):
    QApairs = []
    subject = str(clause.subject)
    directObject = str(clause.direct_object)
    complement = str(clause.complement)
    if answer in subject:
        try:
            QApairs = svoc_subject(clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoc_subject in svoc (generatorRules.py): %s", e)
    elif answer in directObject:
        try:
            QApairs = svoc_dirObject(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoc_dirObject in svoc (generatorRules.py): %s", e)
    elif answer in complement:
        try:
            QApairs = svoc_complement(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoc_complement in svoc (generatorRules.py): %s", e)
    else:
        for j in range(len(clause.adverbials)):
            adverbial = str(clause.adverbials[j])
            if answer in adverbial:
                try:
                    QApairs = svoc_adverbial(verbTag, clause, answer, label)
                except Exception as e:
                    logger.error("Exception in function svoc_adverbial in svoc (generatorRules.py): %s", e)
    return QApairs


def svoc_subject(clause, answer, label):
    QApairs = []
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoc_subject (generatorRules.py): %s", e)
    question = " ".join([pronoun, str(clause.verb), str(clause.direct_object), str(clause.complement), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoc_dirObject(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoc_dirObject (generatorRules.py): %s", e)
    try:
        pronoun = questionPronoun(label)  
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoc_dirObject (generatorRules.py): %s", e)   
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.complement), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.complement), " ".join(str(string) for string in clause.adverbials)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoc_dirObject (generatorRules.py): %s", e)   
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.complement), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoc_complement(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoc_complement (generatorRules.py): %s", e)   
    try:
        pronoun = questionPronoun(label)  
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoc_complement (generatorRules.py): %s", e)         
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoc_complement (generatorRules.py): %s", e)         
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoc_adverbial(verbTag, clause, answer, label): 
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoc_adverbial (generatorRules.py): %s", e)
    # pronoun = "How"
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoc_adverbial (generatorRules.py): %s", e)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.direct_object), str(clause.complement)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.direct_object), str(clause.complement)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoc_adverbial (generatorRules.py): %s", e)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.direct_object), str(clause.complement)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])            
    return QApairs


def svoo(verbTag, clause, answer, label):
    QApairs = []
    subject = str(clause.subject)
    directObject = str(clause.direct_object)
    indirectObject = str(clause.indirect_object)
    if answer in subject:
        try:
            QApairs = svoo_subject(clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoo_subject in svoo (generatorRules.py): %s", e)
    elif answer in directObject:
        try:
            QApairs = svoo_dirObject(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoo_dirObject in svoo (generatorRules.py): %s", e)
    elif answer in indirectObject:
        try:
            QApairs = svoo_indObject(verbTag, clause, answer, label)
        except Exception as e:
            logger.error("Exception in function svoo_indObject in svoo (generatorRules.py): %s", e)
    else:
        for j in range(len(clause.adverbials)):
            adverbial = str(clause.adverbials[j])
            if answer in adverbial:
                try:
                    QApairs = svoo_adverbial(verbTag, clause, answer, label)
                except Exception as e:
                    logger.error("Exception in function svoo_adverbial in svoo (generatorRules.py): %s", e)
    return QApairs


def svoo_subject(clause, answer, label):
    QApairs = []
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoo_subject (generatorRules.py): %s", e)
    question = " ".join([pronoun, str(clause.verb), str(clause.direct_object), str(clause.indirect_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoo_dirObject(verbTag, clause, answer, label):
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoo_dirObject (generatorRules.py): %s", e)
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoo_dirObject (generatorRules.py): %s", e)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.indirect_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.indirect_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoo_dirObject (generatorRules.py): %s", e)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.indirect_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoo_indObject(verbTag, clause, answer, label): 
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoo_indObject (generatorRules.py): %s", e)
    # pronoun = "How"
    try:
        pronoun = questionPronoun(label)    
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoo_indObject (generatorRules.py): %s", e)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    elif len(verb[0])  > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoo_indObject (generatorRules.py): %s", e)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.direct_object), " ".join(str(string) for string in clause.adverbials)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def svoo_adverbial(verbTag, clause, answer, label): 
    QApairs = []
    try:
        verb = transformVerb(str(clause.verb))
    except Exception as e:
        logger.error("Exception in function transformVerb in svoo_adverbial (generatorRules.py): %s", e)
    # pronoun = "How"
    try:
        pronoun = questionPronoun(label)
    except Exception as e:
        logger.error("Exception in function questionPronoun in svoo_adverbial (generatorRules.py): %s", e)
    if len(verb[0]) == 1 and verb[1] == 'be':
        question = " ".join([pronoun, verb[0][0], str(clause.subject), str(clause.direct_object), str(clause.indirect_object)]) + "?"
    elif len(verb[0]) > 1 and (verb[1] == 'be' or verb[1] == 'have'):
        question = " ".join([pronoun, verb[0][0], str(clause.subject), verb[0][1], str(clause.direct_object), str(clause.indirect_object)]) + "?"
    else:
        try:
            aux = auxVerb(verbTag, label)
        except Exception as e:
            logger.error("Exception in function auxVerb in svoo_adverbial (generatorRules.py): %s", e)
        question = " ".join([pronoun, aux, str(clause.subject), verb[1], str(clause.direct_object), str(clause.indirect_object)]) + "?"
    if verifyQuestion(question, QApairs) == 0:
        QApairs.append([question, answer])
    return QApairs


def getPosTags(text):
    doc = nlp(text)
    posTokens = []
    posTags = []
    for token in doc:
        posTokens.append(token)
        posTags.append(token.pos_)
    return posTokens, posTags


def transformVerb(text):
    doc = nlp(text)
 
    # VB  --  verb, base form
    # VBD  --  verb, past tense
    # VBG  --  verb, gerund or present participle
    # VBN  --  verb, past participle
    # VBP  --  verb, non-3rd person singular present
    # VBZ  --  verb, 3rd person singular present

    # POS | AUX
    # VBD | did
    # VBP | do
    # VBZ | does

    aux = ""
    if len(doc) == 1:
        token = doc[0]
        return [[str(token)], token.lemma_]
    else:
        token = doc[0]
        if len(doc) == 2 and str(token) == 'been':
            return [["_", text], token.lemma_]
        elif token.lemma_ == 'be' or token.lemma_ == 'have':
            return [[str(token), " ".join(str(string) for string in doc[1:])], token.lemma_]

    return [text, text]


def questionPronoun(label):
    pronoun = 'What' #['EVENT', 'LANGUAGE', 'LAW', 'NORP', 'PRODUCT', 'WORK_OF_ART']
    if label in ['PERSON', 'ORG']:
        pronoun = 'Who'
    elif label in ['QUANTITY','MONEY','CARDINAL', 'PERCENT']:
        pronoun = 'How much/many'
    elif label in ['TIME','DATE', 'ORDINAL']:
        pronoun = 'When'
    elif label in ['GPE', 'FAC', 'LOC']:
        pronoun = 'Where'
    return pronoun


def auxVerb(rootTag, label):
    aux = "do"  #['I','You','We','They']
    if rootTag == 'VBD': #past
        aux = "did"
    elif rootTag == 'VBZ' or label == 'PERSON': #['He','She','It']
        aux = "does"
    return aux


def verifyQuestion(question, QApairs):
    verify = 0
    for pair in QApairs:
        if question == pair[0]:
            verify = 1
            break
    return verify


def sentenceStemmming(sentence):
    ps = PorterStemmer()
    try:
        words = word_tokenize(sentence)
    except Exception as e:
        logger.error("Exception in function word_tokenize in sentenceStemmming (generatorRules.py): %s", e)
    newSentence = ""
    for w in words:
        if newSentence == "":
            newSentence += ps.stem(w)
        else:
            newSentence += " " + ps.stem(w)
    return newSentence


def replaceStemming(dic):
    newDic = {}
    for index in list(dic.keys()):
        newList = []
        for i in range(len(dic[index])):
            newList.append(sentenceStemmming(dic[index][i]))
        newDic[index] = newList
    return newDic


def sentenceStopwordRemoval(sentence):
    stopwordsSpacy = nlp.Defaults.stop_words
    try:
        tokens = word_tokenize(sentence)
    except Exception as e:
        logger.error("Exception in function word_tokenize in sentenceStopwordRemoval (generatorRules.py): %s", e)
    tokensWithoutStopwords = [word for word in tokens if not word in stopwordsSpacy]
    newSentence = (" ").join(tokensWithoutStopwords)
    return newSentence


def removeStopwords(dic):
    newDic = {}
    for index in list(dic.keys()):
        newList = []
        for i in range(len(dic[index])):
            newList.append(sentenceStopwordRemoval(dic[index][i]))
        newDic[index] = newList
    return newDic


def checkMask(question, answer):
    questionSplit = question.split()

    if "much/many" in questionSplit:
        index = questionSplit.index("much/many")
        questionSplit[index] = "[MASK]"
        listUnmasker = unmasker(' '.join(questionSplit) + " " + answer)
        questionSplit[index] = listUnmasker[0]['token_str']
        question = ' '.join(questionSplit)

    else:
        tokenVoted = questionSplit[0]

        tokenOriginal = questionSplit[0]
        questionSplit[0] = "[MASK]"

        onlyQuestion = ' '.join(questionSplit)
        tokenOnlyQuestion = unmasker(onlyQuestion)[0]['token_str']

        questionAndAnswer = ' '.join(questionSplit) + " " + answer
        tokenQA = unmasker(questionAndAnswer)[0]['token_str']

        tokens = [tokenOriginal.lower(), tokenOnlyQuestion, tokenQA]
        dictCount = {item:tokens.count(item) for item in tokens}
        for key in dictCount.keys():
            if dictCount[key] > 1:
                tokenVoted = key
        
        questionSplit[0] = tokenVoted.capitalize()
        question = ' '.join(questionSplit)
    
    return question


def getQuestions(input):
    for i in range(len(input)):
        sentence = input[i]["sentence"]
        for j in range(len(input[i]["answers"])):
            answer = input[i]["answers"][j]["answer"]
            answerLabel = input[i]["answers"][j]["label"]
            input[i]["answers"][j]["questions"] = []
            try:
                QApairs = produceQuestion(sentence, answer, answerLabel)
            except Exception as e:
                logger.error("Exception in function produceQuestion in getQuestions (generatorRules.py): %s", e)
            for pair in QApairs:
                input[i]["answers"][j]["questions"].append(checkMask(pair[0], pair[1]))
    return input
  