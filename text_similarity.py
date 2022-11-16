import nltk
import string
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from difflib import SequenceMatcher

stopwords = nltk.corpus.stopwords.words('portuguese')

def normalize_string(text:str):
    nfkd = unicodedata.normalize('NFKD', text)
    palavraSemAcento = u"".join([c for c in nfkd if not unicodedata.combining(c)])
    word_list = nltk.word_tokenize(palavraSemAcento.lower())
    words =  [word for word in word_list if word not in stopwords]
    words = [word for word in word_list if word not in string.punctuation]
    return " ".join(words)

def cossine_similarity_score(word_list:list):
    cleaned = list(map(normalize_string,word_list))
    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()
    return cosine_similarity(vectors[0].reshape(1,-1),vectors[1].reshape(1,-1))

def sequence_match_score(word_list:list):
    cleaned = list(map(normalize_string,word_list))
    phrase_1 = ' '.join(cleaned[0])
    phrase_2 = ' '.join(cleaned[1])
    return SequenceMatcher(None, phrase_1, phrase_2).ratio()

def similarity_score(word_list:list):
    try:
        cossine = cossine_similarity_score(word_list)[0][0]
    except Exception as e:
        cossine = str(e)
    try:
        sequence = sequence_match_score(word_list)
    except Exception as e:
        sequence = str(e)
    
    if isinstance(cossine, str) or isinstance(sequence, str):
        return 'error - empty string'
    else:
        return ((3*cossine)+sequence)/4


def price_difference(price_list:list):
    try:
        var = (1-(price_list[0]/price_list[1]))
    except:
        var = 1
    if var >= 0.4:
        return 'Price Diff >= 40%'
    if var <= 0.4:
        return 'Price Diff Not Signifficant'