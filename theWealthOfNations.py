import random;
import codecs;
import string;
from nltk.stem.porter import PorterStemmer;
import gensim;
from gensim import corpora;
import numpy as np




stemmer = PorterStemmer() 
random.seed(123) 
f = codecs.open("pg3300.txt", "r", "utf-8")
text = f.read()
f.close()

#TASK 1

# Splits the documents on new paragraphs with double linebreak
def splitOnNewLine(text):
    return text.lower().split("\r\n\r\n")


def removeEmptyElements(list): 
    newList = []
    for i in list: 
        if len(i) != 0: 
            newList.append(i)
    return newList

# Chapters are indented in the file so have to iterate over the list to split on the newline + indentation 
def splitChapters(liste):
    newList = []
    for text in liste: 
        if len(text) == 0: # Removes a paragraph 
            continue 
        temp = text.split("\r\n    \r\n      ")
        if len(temp) == 1: 
            newList.append(temp[0])
        else: 
            for t in temp: 
                newList.append(t)
    return newList

# Turn paragraphs into a list of words 
def tokenizeSplitAndStripText(text):
    tempWordsInList=[]
    wordsInList = []
    listOfLines = text.split("\r\n") # Splits paragraph on linebreak
    for line in listOfLines: # Iterates over lines to split them into individual words
        tempList = line.split(" ")
        for word in tempList:
            tempWordsInList.append(word) 
    for word in tempWordsInList: #Removes empty words and return the individual words in a list
        if len(word) != 0: 
            tempWord = word.strip(string.punctuation+"\n\r\t")
            wordsInList.append(stemmer.stem(tempWord))
    return wordsInList

tempParList = splitOnNewLine(text)
parList = splitChapters(tempParList)


def removeGutenberg(parList): # Removes the paragraphs that contain the word gutenberg
    tempParListWithoutGutenberg=[]
    for par in parList:
        if (not("gutenberg" in par)):
            if len(par) != 0: 
                tempParListWithoutGutenberg.append(par)
    return tempParListWithoutGutenberg


parListWithoutGutenberg= removeGutenberg(parList)

def splitIntoWord(liste): # Iterates over paragraphs to turn them into lists of words + removes paragraphs without any content
    tempListe=[]
    for par in liste:
        tempParListSplitInWord = tokenizeSplitAndStripText(par)
        if len(tempParListSplitInWord) != 0:
            tempListe.append(tempParListSplitInWord) 
    return tempListe
    
    
def removeEmpty(liste): # Removes paragraphs without any content
    tempListe=[]
    for par in liste:
        tempParListSplitInWord = tokenizeSplitAndStripText(par)
        if len(tempParListSplitInWord) != 0:
            tempListe.append(par) 
    return tempListe

        
parListSplitInWords=splitIntoWord(parListWithoutGutenberg)


def stemListOfWords(list): # Takes a list of words and stem them.
    newList=[]
    for word in list: 
        newList.append(stemmer.stem(word))
    return newList

# TASK 2


listOfStopWords = ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']
stemmedListOfStopWords = stemListOfWords(listOfStopWords)
dictionary = gensim.corpora.Dictionary(parListSplitInWords)

stopIds=[]
for word in stemmedListOfStopWords: # Filters out words from dictionary that appear in the stoplist
    if word in dictionary.token2id:  #Checks if the word in the stopwordlist exists in the dictionary | To prevent wrongfull calls
        stopIds.append(dictionary.token2id[word])
dictionary.filter_tokens(stopIds)



def removeStopWordsFromList(list): # Removes stopwords from a list of words | Used to process Queries
    newList=[]
    for word in list: 
        if word not in stemmedListOfStopWords: 
            newList.append(word)
    return newList

new_vec = [dictionary.doc2bow(par) for par in parListSplitInWords]

# Task 3
tfidf_model = gensim.models.TfidfModel(new_vec)
tfidf_corpus = tfidf_model[new_vec]
tfidf_index = gensim.similarities.MatrixSimilarity(tfidf_corpus)

lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[new_vec]
lsi_index = gensim.similarities.MatrixSimilarity(lsi_corpus)


def preprocessing(query): # Processes the query
    tempQue = tokenizeSplitAndStripText(query) # Removes spaces, linebreaks and punctuation and returns a list of words
    que = removeStopWordsFromList(tempQue) # Removes stopwords from a list
    return que


fullParagraphs = removeEmpty(removeGutenberg(splitChapters(splitOnNewLine(text))))

# Task 4
def tfidfRelevantParagraphs(numOfPar, listOfPar, query): # Finds the numOfPar most relevant paragraphs to query in the corpus and prints them. 
    query = preprocessing(query) #Processes the query
    query = dictionary.doc2bow(query) #Puts each word together with the number of words in the query
    tfidf_query = tfidf_model[query] #Puts the query together with the weight of the word
    doc2similarity = enumerate(tfidf_index[tfidf_query]) # Finds similiarity between the documents in the corpus and the query
    sortedList = sorted(doc2similarity, key=lambda kv: (-kv[1]))[:numOfPar] # Sorts by relevance and returns the numOfPar most relevant
    for i in range (len(sortedList)): #Prints the results
        print(i+1, ". Best match")
        print()
        print(listOfPar[sortedList[i][0]])
        print("[----------------------------------]")


def lsiRelevantTopics(numOfTop, query):
    query = preprocessing(query)  #Processes the query
    query = dictionary.doc2bow(query)  #Puts each word together with the number of words in the query
    tfidf_query = tfidf_model[query]  #Puts the query together with the weight of the word
    lsi_query = lsi_model[tfidf_query]  #The relevant topics to the query is set to lsi_query
    sortedTopic = sorted(lsi_query, key=lambda kv: -abs(kv[1]))[:numOfTop]  #Sorts the lsi_queries based on the most relevant, and sets the number of wished queries to sortedTopics
    for i in range (len(sortedTopic)):  #Prints out the most relevant topics from the lsi_query
        print(i+1, ". Best topic match")
        print(lsi_model.show_topics()[sortedTopic[i][0]])
        print("[------------------------------------]")
    
print("------ 3.5 -------")
print(lsi_model.show_topics(3))
print("\r\n")
print("We think it chooses the different words in a topic based in the words that often appear together.")
print("\r\n")
print("------ 4.3 -------")
tfidfRelevantParagraphs(3, fullParagraphs, "What is the function of money?")
print("\r\n")
print("------ 4.4 -------")
lsiRelevantTopics(3, "What is the function of money?")
