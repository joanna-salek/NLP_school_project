from data_load import loading_from_scratch


def preprocessing():
    # preprocessing
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import string
    data = []
    movies = loading_from_scratch()
    stop_words = set(stopwords.words("english"))
    for review in movies:
        sentence = word_tokenize(review[0].lower())
        sentence = [x for x in sentence if x not in stop_words and x not in string.punctuation
                    and x not in ("i", "'s", "''", "n't", "``", "...", "<br />")]
        sentence = tuple(sentence)
        data.append((sentence, review[1]))
    return data


def words():
    # save words to document
    document = []
    for review in preprocessing():
        for word in review[0]:
            document.append(word)
    return document
