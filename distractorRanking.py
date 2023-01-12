""" This file includes functions to sort distractors according to BERT, GPT2, spaCy and FitBERT, as well as
methods to evaluate and compare these approaches (position difference and top 5 accuracy). """

import spacy
import torch
from transformers import BertTokenizer, BertForNextSentencePrediction
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from fitbert import FitBert
nlp = spacy.load('en_core_web_lg')
tokenizerBERT = BertTokenizer.from_pretrained("bert-base-uncased")
modelBERT = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
tokenizerGPT2 = GPT2Tokenizer.from_pretrained("gpt2")
modelGPT2 = GPT2LMHeadModel.from_pretrained("gpt2")
fb = FitBert()
from loggerLog import *


""" Get BERT score (loss) based on Next Sentence Prediction.
In the case of distractors, the first sentence is the question and the second sentence is one of the
distractors. """
def scoreBERT(tokenizer, model, firstSentence, secondSentence):
    encoding = tokenizer(firstSentence, secondSentence, return_tensors="pt")
    output = model(**encoding, labels=torch.LongTensor([1]))
    loss = output.loss
    logits = output.logits
    return output.loss.item()


""" Get GPT2 score (loss) based on the probability of the question and a distractor being together.
The input is a single string "question + distractor". """
def scoreGPT2(tokenizer, model, text):
    input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
    output = model(input_ids, labels=input_ids)
    loss = output.loss
    logits = output.logits
    return loss.item()


""" Each word is transformed in a vector, with the value correspondent to a spaCy document being the
average of the values of its words (vector "fast food" is the average of the vectors "fast" and "food").
The similarity between documents is calculated with these averages.   
In this function we return the similarity between distractor and question, distractor and answer and the
mean of these two results. """
def spacySimilarity(question, answer, distractor):
    questionNLP = nlp(question)
    answerNLP = nlp(answer)
    distractorNLP = nlp(distractor)

    questionNLP_NoSW = nlp(' '.join([str(t) for t in questionNLP if not t.is_stop]))
    answerNLP_NoSW = nlp(' '.join([str(t) for t in answerNLP if not t.is_stop]))
    distractorNLP_NoSW = nlp(' '.join([str(t) for t in distractorNLP if not t.is_stop]))

    distractorQuestionSimilarity = 0
    distractorAnswerSimilarity = 0

    if(distractorNLP_NoSW and distractorNLP_NoSW.vector_norm):
        if(questionNLP_NoSW and questionNLP_NoSW.vector_norm):
            distractorQuestionSimilarity = distractorNLP_NoSW.similarity(questionNLP_NoSW)
        if(answerNLP_NoSW and answerNLP_NoSW.vector_norm):
            distractorAnswerSimilarity = distractorNLP_NoSW.similarity(answerNLP_NoSW)

    mean = (distractorQuestionSimilarity + distractorAnswerSimilarity) / 2
    
    return distractorQuestionSimilarity, distractorAnswerSimilarity, mean
    

""" Returns dictionary with all distractors sorted by their BERT score (ascendant order, as scores are
loss values, and so smaller values are preferable). """
def sortDistractorsBERT(question, distractors):
    scores = {}
    for distractor in distractors:
        try:
            scores[distractor] = scoreBERT(tokenizerBERT, modelBERT, question, distractor)
        except Exception as e:
            logger.error("Exception in function scoreBERT in sortDistractorsBERT (distractorRanking.py): %s", e)
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    return scores

""" Returns dictionary with all distractors sorted by their GPT2 score (ascendant order, as scores are
loss values, and so smaller values are preferable). """
def sortDistractorsGPT2(question, distractors):
    scores = {}
    for distractor in distractors:
        try:
            scores[distractor] = scoreGPT2(tokenizerGPT2, modelGPT2, question+" "+distractor)
        except Exception as e:
            logger.error("Exception in function scoreGPT2 in sortDistractorsGPT2 (distractorRanking.py): %s", e)
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1])}
    return scores


""" Returns dictionary with all distractors sorted by their spaCy similarity (descendant order, as higher
values represent more similar strings). """
def sortDistractorsSpacy(question, answer, distractors):
    scores = {}
    for distractor in distractors:
        try:
            distractorQuestionSimilarity, distractorAnswerSimilarity, mean = spacySimilarity(question, answer, distractor)
        except Exception as e:
            logger.error("Exception in function spacySimilarity in sortDistractorsSpacy (distractorRanking.py): %s", e)
        scores[distractor] = mean
    scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
    return scores


""" Use of a masked language model to determine how likely it is to a distractor to follow a question,
that is, making sense in the context of the question.
Returns a dictionary with distractors sorted from the better evaluated to the worst. """
def sortDistractorsFitBERT(question, distractors):
    masked_string = question + " ***mask***"
    options = distractors
    ranked_options = fb.rank(masked_string, options=options)

    dict = {}
    for i in range(len(ranked_options)):
        dict[ranked_options[i]] = i+1

    return dict


def calculateDifference(ref, hyp, reverseFlag):
    try:
        ref = indexesDict(ref, True)
        hyp = indexesDict(hyp, reverseFlag)
    except Exception as e:
        logger.error("Exception in function indexesDict in calculateDifference (distractorRanking.py): %s", e)
    dif = 0
    for (key, value) in ref.items():
        dif += abs(value - hyp[key])
    return dif


def indexesDict(dict, reverseFlag):
    dictValues = list(set(dict.values()))
    dictValues.sort(reverse=reverseFlag)
    dictPos = {}
    for (key, value) in dict.items():
        dictPos[key] = dictValues.index(value)+1
    return dictPos


"""
Given a certain number of sets (each composed by the original sentence, question generated, answer and
distractors), compare the results obtained with each ranking method with the ranking obtained with
potential users answering to forms.
In this case, we compare the position each distractors is in the hypotesis (ranking method) and in the
reference (people opinion) and sum the differences of these comparisons.
"""
def totalDifference(sets, model):
    totalDif = 0
    for set in sets:
        question = set[1]
        answer = set[2]
        distractors = set[3]
        hypothesis = []
        if model=="bert":
            try:
                hypothesis = sortDistractorsBERT(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsBERT in totalDifference (distractorRanking.py): %s", e)
            try:
                totalDif += calculateDifference(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function calculateDifference in totalDifference (distractorRanking.py): %s", e) 
        elif model=="gpt2":
            try:
                hypothesis = sortDistractorsGPT2(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsGPT2 in totalDifference (distractorRanking.py): %s", e)
            try:
                totalDif += calculateDifference(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function calculateDifference in totalDifference (distractorRanking.py): %s", e)
        elif model=="spacy":
            try:
                hypothesis = sortDistractorsSpacy(question, answer, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsSpacy in totalDifference (distractorRanking.py): %s", e)
            try:
                totalDif += calculateDifference(distractors, hypothesis, True)
            except Exception as e:
                logger.error("Exception in function calculateDifference in totalDifference (distractorRanking.py): %s", e)
        elif model=="fitbert":
            try:
                hypothesis = sortDistractorsFitBERT(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsFitBERT in totalDifference (distractorRanking.py): %s", e)
            try:
                totalDif += calculateDifference(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function calculateDifference in totalDifference (distractorRanking.py): %s", e)
        
    return totalDif


def top5accuracy(ref, hyp, reverseFlag):
    correct = 0

    maxLen = 5
    if len(hyp) < maxLen: maxLen = len(hyp)
    if len(ref) < maxLen: maxLen = len(ref)
    
    try:
        hypTopKeys = getTopKeys(hyp, maxLen, reverseFlag)
        refTopKeys = getTopKeys(ref, maxLen, True)
    except Exception as e:
        logger.error("Exception in function getTopKeys in top5accuracy (distractorRanking.py): %s", e)
    
    for hypTopKey in hypTopKeys:
        if hypTopKey in refTopKeys:
            correct += 1
    
    return correct/maxLen


def getTopKeys(dict, maxLen, reverseFlag):
    dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=reverseFlag)}

    lastValue = None
    dictTop = {}
    for (key, value) in dict.items():
        if len(dictTop) >= maxLen and lastValue != value:
            return list(dictTop.keys())
        dictTop[key] = value
        lastValue = value
    return dictTop.keys()


"""
Another way to evaluate the performance of the ranking methods.
We limit to the 5 best evaluated distractors in reference and in hypothesis and verify how many of them
are present in both lists.
"""
def totalTop5accuracy(sets, model):
    totalResult = 0
    for set in sets:
        question = set[1]
        answer = set[2]
        distractors = set[3]
        hypothesis = []
        if model=="bert":
            try:
                hypothesis = sortDistractorsBERT(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsBERT in totalTop5accuracy (distractorRanking.py): %s", e)
            try:
                totalResult += top5accuracy(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function top5accuracy in totalTop5accuracy (distractorRanking.py): %s", e)
        elif model=="gpt2":
            try:
                hypothesis = sortDistractorsGPT2(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsGPT2 in totalTop5accuracy (distractorRanking.py): %s", e)
            try:
                totalResult += top5accuracy(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function top5accuracy in totalTop5accuracy (distractorRanking.py): %s", e)
        elif model=="spacy":
            try:
                hypothesis = sortDistractorsSpacy(question, answer, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsSpacy in totalTop5accuracy (distractorRanking.py): %s", e)
            try:
                totalResult += top5accuracy(distractors, hypothesis, True)
            except Exception as e:
                logger.error("Exception in function top5accuracy in totalTop5accuracy (distractorRanking.py): %s", e)
        elif model=="fitbert":
            try:
                hypothesis = sortDistractorsFitBERT(question, distractors)
            except Exception as e:
                logger.error("Exception in function sortDistractorsFitBERT in totalTop5accuracy (distractorRanking.py): %s", e)
            try:
                totalResult += top5accuracy(distractors, hypothesis, False)
            except Exception as e:
                logger.error("Exception in function top5accuracy in totalTop5accuracy (distractorRanking.py): %s", e)
    return totalResult/len(sets)


if __name__ == '__main__':
    """
    Sentences chosen from Wikipedia articles.
    Questions, answers and distractors generated or selected with the developed approaches.
    The distractors were evaluated resorting to human opinion. The dictionaries contain the distractors
    sorted from most to less chosen. Keys with the same value were chosen an equal number of times.
    """
    sentence1 = "Coimbra is a city and a municipality in Portugal."
    question1 = "In what country is Coimbra located?"
    answer1 = "Portugal"
    distractors1 = {"Lisbon":6, "Algarve":5, "Brazil":10, "France":9, "Porto":5, "Spain":10, "Greece":10, "Portuguese":4,
        "Italy":10, "Argentina":9, "Poland":9, "Coimbra":6, "Portuguese Province":3, "Portuguese City":3}
    
    sentence2 = "About 460,000 people live in Coimbra, comprising 19 municipalities and extending into an area of 4,336 square kilometres (1,674 sq mi)."
    question2 = "What is the area of Coimbra?"
    answer2 = "4,336 square kilometres"
    distractors2 = {"73 meters":5, "920 metres":4, "1,674 sq mi":2, "10 hectares":9, "3,018 feet":5, "metre":4, "railway yard":4,
        "plot of land":4, "plot of ground":4, "picnic area":4, "4,336 square kilometers":1, "4,336 square miles":9, "4,336 square 160":6,
        "3,000 square kilometers":7, "3,400 square kilometers":9, "3,732 square kilometers":9, "3,732 kilometers":5}

    sentence3 = "Star Wars (retroactively titled Star Wars: Episode IV - A New Hope) is a 1977 American epic space opera film written and directed by George Lucas, produced by Lucasfilm and distributed by 20th Century Fox."
    question3 = "Who directed the 1977 Star Wars film?"
    answer3 = "George Lucas"
    distractors3 = {"Peter Jackson":9, "Michael Moore":8, "w. lucas":7, "The Star Wars":4, "The Star Wars Star":4, "Star Wars":4,
        "Mark Hamill":7, "william lucas":6, "charles lucas":7, "howard lucas":6, "john lucas":7, "Peter Cooper":8, "Robert E. Howard":8,
        "Jason Scott":7, "James Taylor":7, "John Williams":8, "Martin Smith":7, "Charles":5, "John Simon":7, "John Carter":7}

    sentence4 = "When adjusted for inflation, Star Wars is the second-highest-grossing film in North America (behind Gone with the Wind) and the fourth-highest-grossing film of all time."
    question4 = "Where does Star Wars rank in grossing movies in North America?"
    answer4 = "second"
    distractors4 = {"10th":5, "third":8, "first":10, "Star Wars":4, "Star Wars first":4, "sixth":8, "fifth":7, "fourth":10, "eighth":6}

    sentence5 = "The latter featured \"Bohemian Rhapsody\", which stayed at number one in the UK for nine weeks and helped popularise the music video format."
    question5 = "What song was featured in the video?"
    answer5 = "Bohemian Rhapsody"
    distractors5 = {"Rhapsody":3, "beatnik":5, "Czech":4, "moravian rhapsody":6, "bohemia rhapsody":6, "Queen":6, "Queen in London":5,
    "British rock":6, "rock band":4, "British":4, "bohemian serenade":8, "transylvanian rhapsody":7, "aristocratic rhapsody":6, "Austrian":4, 
    "Grecian":4, "Hungarian":4, "Queen + Wyclef Jean":4, "Queen released Jazz":4, "Seaside Rendezvous":6, "John Lennon Ealing College":4}

    sentence6 = "Queen are a British rock band formed in London in 1970."
    question6 = "When was Queen formed?"
    answer6 = "1970"
    distractors6 = {"1974":8, "British":4, "Before London":4, "1971":11, "Before 1970":8, "By London":4,
        "After London":4, "1977":8, "1969":8, "1965":7, "1966":8, "1968":7, "1972":9, "1973":8}

    sentence7 = "Europe is bordered by the Arctic Ocean to the north, the Atlantic Ocean to the west, the Mediterranean Sea to the south and Asia to the east."
    question7 = "What ocean borders Europe to the north?"
    answer7 = "the Arctic Ocean"
    distractors7 = {"the Gulf Stream":8, "sea":4, "Geography of North America":4, "the arctic sea":5, "China and Europe":4, 
        "China":4, "the Middle Pacific":9, "the Atlantic Ocean":10, "the Black Sea":8, "the arctic atlantic":5, "the arctic coast":5,
        "the arctic seas":3, "the arctic waters":4, "Environment of North America":4, "Flora of Northern Europe":4, 
        "Shipwrecks of North Asia":4, "Flora of Eastern Europe":4, "territorial waters":5, "seven seas":4, "international waters":4, 
        "high sea":5, "North Atlantic Drift":4, "the Northern Hemisphere":4, "the Emba River":4, "the Southern Hemisphere":5}

    sentence8 = "In 1949, the Council of Europe was founded with the idea of unifying Europe to achieve common goals and prevent future wars."
    question8 = "What organization was founded in 1949?"
    answer8 = "the Council of Europe"
    distractors8 = {"the Tsardom of Russia":7, "Organization of American States":5, "Government by continent":4, "the commission of europe":10,
        "the council of european":3, "Europe's umbrella of Europe":5, "Europe's continent":6, "Europe's branch of Europe":5, "Europeans":4, 
        "the council of america":6, "the council of asia":6, "the council of countries":6, "Lists of continents":4, 
        "Personifications of continents":4, "Buildings and structures by continent":4, "History by continent":4,
        "Commonwealth of Independent States":7, "League of Nations":8, "Organization for the Prohibition of Chemical Weapons":7,
        "United Nations agency":8, "the European Capital of Sport":4, "the European Capital of Culture":4, "the Union of Krewo":5,
        "the European Theatre of World War II":4}

    sentence9 = "Cristiano Ronaldo dos Santos Aveiro (born 5 February 1985) is a Portuguese professional footballer who plays as a forward for Premier League club Manchester United and captains the Portugal national team."
    question9 = "What nationality is Cristiano Ronaldo?"
    answer9 = "Portuguese"
    distractors9 = {"Spanish":10, "Footballers":4, "English":10, "English and Brazilian":5, "English player":5, "Portuguese":1, "argentine":4,
        "italian":4, "portugal":4, "brazilian":9, "Romanian":8, "Catalan":8, "Norwegian":7, "Italian":8, "Dutch":6, "Argentine-Spanish":7}

    sentence10 = "Often considered the best player in the world and  widely regarded as one of the greatest players of all time, Cristiano Ronaldo has won five Ballon d'Or awards and four European Golden Shoes, the most by a European player."
    question10 = "How many Ballon d'Or awards has Cristiano Ronaldo won?"
    answer10 = "five"
    distractors10 = {"six":9, "seven":9, "seven awards":6, "seven games":4, "seven cups":5, "three":8, "four":11, "eight":7}
    
    sets = [
        [sentence1, question1, answer1, distractors1],
        [sentence2, question2, answer2, distractors2],
        [sentence3, question3, answer3, distractors3],
        [sentence4, question4, answer4, distractors4],
        [sentence5, question5, answer5, distractors5],
        [sentence6, question6, answer6, distractors6],
        [sentence7, question7, answer7, distractors7],
        [sentence8, question8, answer8, distractors8],
        [sentence9, question9, answer9, distractors9],
        [sentence10, question10, answer10, distractors10]
    ]

    #print("bert: ", sortDistractorsBERT(question1, distractors1), "\n")
    #print("gpt2: ", sortDistractorsGPT2(question1, distractors1), "\n")
    #print("spacy: ", sortDistractorsSpacy(question1, answer1, distractors1), "\n")
    #print("fitbert: ", sortDistractorsFitBERT(question1, distractors1), "\n")

    
    print("\n---- POSITION DIFFERENCE -----")
    print("bert: ", totalDifference(sets, "bert"))
    print("gpt2: ", totalDifference(sets, "gpt2"))
    print("spacy: ", totalDifference(sets, "spacy"))
    print("fitbert: ", totalDifference(sets, "fitbert"))

    print("\n---- TOP 5 ACCURACY -----")
    print("bert: ", totalTop5accuracy(sets, "bert"))
    print("gpt2: ", totalTop5accuracy(sets, "gpt2"))
    print("spacy: ", totalTop5accuracy(sets, "spacy"))
    print("fitbert: ", totalTop5accuracy(sets, "fitbert"))