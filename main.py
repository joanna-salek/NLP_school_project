import random

import nltk

from most_common import *
from pickle_load import *

document = load_document()
reviews = load_reviews()
m_common = load_mcommon()


def freq():
    # counting frequency of all words
    import matplotlib.pyplot as plt
    print(f"There are {len(document)} words")
    print(f"There are {len(set(document))} unique words")
    fdist().plot(40, title='Plot of 40 most popular words')
    plt.show()


def find_features(doc):
    words = set(doc)
    features = {}
    for w in m_common[:300]:
        # True or False, for word in review if word in most common
        features[w] = (w in words)
    return features


def featureset(x):
    # saves list of tuples find feature + category
    return [(find_features(rev), category) for (rev, category) in x]


# divides reviews into training and test
def training(reviews):
    return featureset(reviews)[:40000]


def test(reviews):
    return featureset(reviews)[40000:50000]


def N_B():
    random.shuffle(reviews)
    training_set = training(reviews)
    testing_set = test(reviews)
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    classifier.show_most_informative_features(15)  # 15 most important words
    y = (nltk.classify.accuracy(classifier, testing_set)) * 100
    print(f"Naive Bayes accuracy on test set: {y}")
    return classifier


def regresion():
    from nltk.classify.scikitlearn import SklearnClassifier
    from sklearn.linear_model import LogisticRegression
    random.shuffle(reviews)
    training_set = training(reviews)
    testing_set = test(reviews)
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    y = (nltk.classify.accuracy(LogisticRegression_classifier, testing_set)) * 100
    print(f"Logistic Regression accuracy on test set: {y}")
    return LogisticRegression_classifier


def SVC():
    from sklearn.svm import LinearSVC
    from nltk.classify.scikitlearn import SklearnClassifier
    random.shuffle(reviews)
    training_set = training(reviews)
    testing_set = test(reviews)
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    y = (nltk.classify.accuracy(LinearSVC_classifier, testing_set)) * 100
    print(f"Linear SVC accuracy on test set: {y}")
    return LinearSVC_classifier


from nltk.classify import ClassifierI
from statistics import mode


class AggClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf


def all_classyfires():
    agg_classifier = AggClassifier(N_B(), regresion(), SVC())
    random.shuffle(reviews)
    testing_set = test(reviews)
    y = (nltk.classify.accuracy(agg_classifier, testing_set)) * 100
    print(f"Combine classifier accuracy on test set: {y}")
    return y


def compare_accuracy():
    # Naive bayes
    for i in range(10):
        print(N_B())

    # Linear regression
    for i in range(10):
        print(regresion())

    # SVM 10 times
    for i in range(10):
        print(SVC())

    # all classifiers
    for i in range(10):
        print(all_classyfires())
