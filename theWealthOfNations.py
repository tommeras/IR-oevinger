import random;
import codecs;
import string;
from nltk.stem.porter import PorterStemmer;
import gensim;
from gensim import corpora;




stemmer = PorterStemmer() 
random.seed(123) 
f = codecs.open("pg3300.txt", "r", "utf-8")
text = f.read()
f.close()

#TASK 1

#f.split("\n")
def splitOnNewLine(text):
    return text.lower().split("\r\n\r\n")
def splitChapters(liste):
    newList = []
    for text in liste: 
        if len(text) == 0: 
            continue 
        temp = text.split("\r\n    \r\n      ")
        if len(temp) == 1: 
            newList.append(temp[0])
        else: 
            for t in temp: 
                newList.append(t)
    return newList

def tokenizeSplitAndStripText(text):
    tempWordsInList=[]
    wordsInList = []
    listOfLines = text.split("\r\n")
    for line in listOfLines:
        tempList = line.split(" ")
        for word in tempList:
            tempWordsInList.append(word) 
    for word in tempWordsInList: 
        if len(word) != 0: 
            tempWord = word.strip(string.punctuation+"\n\r\t")
            wordsInList.append(stemmer.stem(tempWord))
    return wordsInList

tempParList = splitOnNewLine(text)
parList = splitChapters(tempParList)
parListWithoutGutenberg=[]
for par in parList:
    parLower = par.lower()
    if (not("gutenberg" in parLower)):
        parListWithoutGutenberg.append(par)
parListSplitInWords=[]

for par in parListWithoutGutenberg:
    tempParListSplitInWord = tokenizeSplitAndStripText(par)
    if len(tempParListSplitInWord) != 0:
        parListSplitInWords.append(tempParListSplitInWord) 


def stemListOfWords(list): 
    newList=[]
    for word in list:
        newList.append(stemmer.stem(word))
    return newList

listOfStopWords = ['a','able','about','across','after','all','almost','also','am','among','an','and','any','are','as','at','be','because','been','but','by','can','cannot','could','dear','did','do','does','either','else','ever','every','for','from','get','got','had','has','have','he','her','hers','him','his','how','however','i','if','in','into','is','it','its','just','least','let','like','likely','may','me','might','most','must','my','neither','no','nor','not','of','off','often','on','only','or','other','our','own','rather','said','say','says','she','should','since','so','some','than','that','the','their','them','then','there','these','they','this','tis','to','too','twas','us','wants','was','we','were','what','when','where','which','while','who','whom','why','will','with','would','yet','you','your']
stemmedListOfStopWords = stemListOfWords(listOfStopWords)
dictionary = gensim.corpora.Dictionary(parListSplitInWords)
stopIds=[]
for word in stemmedListOfStopWords:
    if word in dictionary.token2id:  
        stopIds.append(dictionary.token2id[word])
dictionary.filter_tokens(stopIds)
new_vec = [dictionary.doc2bow(par.lower().split()) for par in parList]


def removeStopWordsFromList(list): 
    newList=[]
    for word in list: 
        if word not in stemmedListOfStopWords: 
            newList.append(word)
    return newList

tfidf_model = gensim.models.TfidfModel(new_vec)
print(tfidf_model)
tfidf_corpus = tfidf_model[new_vec]
print(tfidf_corpus[1])
index = gensim.similarities.MatrixSimilarity(tfidf_corpus)
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[new_vec]
lsiIndex = gensim.similarities.MatrixSimilarity(lsi_corpus)


def preprocessing(query): 
    tempQue = tokenizeSplitAndStripText(query)
    que = removeStopWordsFromList(tempQue)
    return que
query = preprocessing("What is the function of money?")
print(query)
query = dictionary.doc2bow(query) 
print(query)