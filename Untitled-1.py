from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer() 
word = "Any" 
print(stemmer.stem(word.lower())) 
