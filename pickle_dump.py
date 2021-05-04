import pickle
from preprocessing import *
from most_common import most_common


def save_words():
    with open("document.txt", "wb") as f:
        pickle.dump(words(), f)

def save_reviews():
    with open("recenzje.txt", "wb") as f:
        pickle.dump(preprocessing(), f)

def save_most_common():
    with open("most_common.txt", "wb") as f:
        pickle.dump(most_common(3000), f)

