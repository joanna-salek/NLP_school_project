import pickle


def load_document():
    # loading document containing all words to document
    with open("document.txt", "rb") as f:
        document = pickle.load(f)
    return document


def load_reviews():
    # loading movie reviews from pickle
    with open("recenzje.txt", "rb") as f:
        reviews = pickle.load(f)[1:]
    return reviews


def load_mcommon():
    # loading most common words
    with open("most_common.txt", "rb") as f:
        m_common = pickle.load(f)
    return m_common
