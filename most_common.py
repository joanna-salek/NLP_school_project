from pickle_load import load_document

def fdist():
    # counting frequency of all words
    from nltk.probability import FreqDist
    document = load_document()
    return FreqDist(document)


def most_common(x):
    # returns most common x words in document
    return [z[0] for z in fdist().most_common(x)]
