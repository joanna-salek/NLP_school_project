import random

import nltk

from most_common import fdist
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


def find_features(document):
    words = set(document)
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


def test(rewievs):
    return featureset(rewievs)[40000:50000]


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
        print (SVC())

    # all classifiers
    for i in range(10):
        print (all_classyfires())


# Neural network


import torch
import torch.nn as nn

random.shuffle(reviews)
data = reviews[:40000]

test_data = reviews[40000:]

together = data + test_data

label_to_ix = {"0": 0, "1": 1}

word_to_ix = {}  # tutaj wrzucamy wszystkie slowa, kazde ma indywidualny numer (kolejna liczba naturalna)
for word in m_common(300):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(label_to_ix)

print(VOCAB_SIZE, NUM_LABELS)
print(word_to_ix)


def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        if word in word_to_ix:
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1)


def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])  # https://pytorch.org/docs/stable/tensors.html


import torch.nn.functional as F  # tu są rozne funkcje aktywacji


class BoWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super().__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec), dim=1)


model = BoWClassifier(NUM_LABELS, VOCAB_SIZE)

print(list(model.parameters()))

import torch.autograd as autograd

loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

n_iters = 100
for epoch in range(n_iters):
    for instance, label in data:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))

        # forward
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)

        # backward
        loss.backward()
        optimizer.step()

        # zerujemy gradient
        optimizer.zero_grad()

z = 0
m = 0
for instance, label in test_data:
    m += 1
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    if label_to_ix[label] == torch.argmax(log_probs):
        z += 1
print("Dokładność to ", z / m)

print(list(model.parameters()))

import torchtext

glove = torchtext.vocab.GloVe(name="6B", dim=200)

import torch.nn as nn
import torch


def get_review_vectors(glove_vector):
    train, valid, test = [], [], []  # tworze trzy listy na dane train, valid i test
    for i, rewiev in enumerate(reviews):
        if i % 59 == 0:
            re_emb = sum(glove_vector[w] for w in rewiev[0])
            label = torch.tensor(int(rewiev[-1] == "1")).long()  # generuje dwa rodzaje labelow: 1 - gdy pos, 0 gdy neg

            # dzielimy dane na trzy kategorie
            if i % 20 < 14:  # czyli 0,1,2,3,4,5,6,7,8,9,10,11,12,13
                train.append((re_emb, label))  # 70% danych treningowych
            elif i % 20 in (14, 15, 16):  # czyli 14, 15, 16
                valid.append((re_emb, label))  # 15% danych do walidacji
            else:  # czyli gddy i%20 jest 17, 18, 19
                test.append((re_emb, label))  # 15% danych testowych
    return train, valid, test


train, valid, test = get_review_vectors(glove)  # przygotowuje sobie dane w oparciu o gotowe embeddingi z glove

# za kazdym razem będzie bral 128 rekordow (przy trenowaniu) i za kazdym razem będą to inne losowe rekordy (shuffle = True)
train_loader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)


# trenowanie
def train_network(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-5):
    criterion = nn.CrossEntropyLoss()  # funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optymalizator ADAM
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):  # dla kazdej epoki
        for re, labels in train_loader:  # przechodze dane treningowe
            optimizer.zero_grad()
            pred = model(re)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))  # zapisuje wartosc funkcji kosztu
        if epoch % 20 == 0:  # co 20 epoke
            epochs.append(epoch)  # zapisz numer epoki
            train_acc.append(get_accuracy(model, train_loader))  # dokladnosc na zbiorze testowym
            valid_acc.append(get_accuracy(model, valid_loader))  # dokladnosc na zbiorze treningowym
            print(
                f'Epoch number: {epoch + 1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1], 3)} | Valid accuracy: {round(valid_acc[-1], 3)}')
    # Rysowanie wykresow
    plt.title("Training Curve")
    plt.plot(losses, label="Train dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train dataset")
    plt.plot(epochs, valid_acc, label="Validation dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


# Funkcja wyznaczająca dokładność predykcji:
def get_accuracy(model, data_loader):
    correct, total = 0, 0  # ile ok, ile wszystkich
    for tweets, labels in data_loader:  # przechodzi dane
        output = model(tweets)  # jak dziala model
        pred = output.max(1, keepdim=True)[1]  # ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total


# Siec neuronową moge zapisac w taki uproszczony spoosob
mymodel = nn.Sequential(nn.Linear(200, 100),
                        nn.GELU(),
                        nn.Linear(100, 2))

train_network(mymodel, train_loader, valid_loader, num_epochs=1000, learning_rate=0.0000001)

print("Final test accuracy:", get_accuracy(mymodel, test_loader))  # dokladnosc na zbiorze testowym


def test_model(model, glove_vector, re):
    emb = sum(glove_vector[w] for w in re)  # przerabiam tweet na sume embieddingpw
    out = mymodel(emb.unsqueeze(0))  # co powie model
    pred = out.max(1, keepdim=True)[1]  # ktora kategoria bardziej prawdopodobna
    return pred


import torch
import torch.nn as nn
import torchtext
import matplotlib.pyplot as plt

# SIECI NEURONOWE + GRU/LSTM

glove = torchtext.vocab.GloVe(name="6B", dim=200,
                              max_vectors=10000)  # 50 wymiarowe wekotry, uzywamy 10 tys najczęstszych slow


def get_tweet_words(glove_vector):  # argumenty: glove_vector - info o embeddingach, data = dane
    train, valid, test = [], [], []  # puste listy na dane treningowe, walidacyjne, do testowania
    for i, line in enumerate(reviews):
        if i % 29 == 0:  # wybieram co 29 rekord (zeby w miare szybko sie nauczyl model; normalnie by mozna pominąc tego ifa)
            tweet = line[0]  # kolejny tweet
            idx = [glove_vector.stoi[w] for w in tweet if
                   w in glove_vector.stoi]  # zapisuje indeksy slow ktore mialy embeddingi
            if not idx:  # jezeli zdarzy sie jakis tweet ktory nie ma zadnego embeddingu to pobiera kolejny rekord (nie robi tego co dalej)
                continue
            idx = torch.tensor(idx)  # zapisuje indeksy jakot tensor
            label = torch.tensor(int(line[1] == "0")).long()
            # label dla tweeta, 0 gdy bylo 0, 1 gdy  bylo 4 (wartosc logiczną tranformuje w liczbe calkowitą a następnie zapisuje do tensora)
            # chce sobie podzielic dane na trzy kategorie: zbior treningowy, walidacyjny i testowy [3 na 5 będe zapisywal do testowego, 1 na 5 do walidacyjnego, 1 na 5 do testowego]
            if i % 20 < 14:  # czyli 0,1,2 [60% danych]
                train.append((idx, label))
            elif i % 20 in (14, 15, 16):  # 20 % danych
                valid.append((idx, label))
            else:  # pozostale 20% danych
                test.append((idx, label))
    return train, valid, test  # zwraca zbior treningowy, walidacyjny i testowy


train, valid, test = get_tweet_words(glove)

tweet0, label0 = train[0]

glove_emb = nn.Embedding.from_pretrained(glove.vectors)  # zeby uzyc potem gotowych embeddingow
tweet_emb = glove_emb(tweet0)  # tworzy mi tensor z embeddingami dla kazdego slowa z tweet0

tweet_input = tweet_emb.unsqueeze(0)  # dodaje dodatkowy wymiar (bo taki będzie oczekiwany pozniej)

for i in range(10):
    tweet, label = train[i]

rnn_layer = nn.RNN(input_size=200,  # wymiar cech wejscia (wymiar embeddinga)
                   hidden_size=5,  # wymiar cech w stanie ukrytym
                   batch_first=True)

h0 = torch.zeros(1, 1, 5)

out, last_hidden = rnn_layer(tweet_input, h0)
out2, last_hidden2 = rnn_layer(tweet_input)


class T_RNN(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes):  # parametry: input_size = wymiar Embeddingu, hidden_size - zadajemy, num_classes = ile kategorii
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors)  # embeddingi
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # RNN
        self.fc = nn.Linear(hidden_size, num_classes)  # przeksztalcenie liniowe

    def forward(self, x):
        x = self.emb(x)  # wyznacza embedding
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # początkowy stan ukryty
        out, _ = self.rnn(x, h0)  # RNN
        out = self.fc(out[:, -1, :])  # przez funkcje liniową przekształcamy ostatni output
        return out


model = T_RNN(200, 6, 2)

import random


class TBatcher:
    def __init__(self, tweets, batch_size=1, drop_last=False):
        self.tweets_by_length = {}  # slownik, przechowuje klucze - dlugosci i wartosci - liste tweetow o zadanej dlugosci
        for words, label in tweets:

            wlen = words.shape[0]

            if wlen not in self.tweets_by_length:  # jak jeszcze nie pojawil sie tweet o takiej dlugosci
                self.tweets_by_length[wlen] = []  # to stworz go i przypisz mu pustą liste

            self.tweets_by_length[wlen].append((words, label), )  # dodaje do listy krotke slowa, label
            # tworze DataLoader dla kazdego zbioru tweetow o tej samej dlugosci
        self.loaders = {
            wlen: torch.utils.data.DataLoader(tweets, batch_size=batch_size, shuffle=True, drop_last=drop_last) for
            wlen, tweets in self.tweets_by_length.items()}

    # Iterator, to nie takie wazne...
    def __iter__(self):
        iters = [iter(loader) for loader in self.loaders.values()]  # tworze iterator dla kazdej dlugosci tweetow
        while iters:
            im = random.choice(iters)  # generuje losowy iterator
            try:
                yield next(
                    im)  # yield uzywamy kiedy iterujemy po sekwencji ale nie chcemy przechowywac calej sekwencji w pamieci (cos jak return)
            except StopIteration:
                iters.remove(im)


train_TB = TBatcher(train, drop_last=True)

for i, (tweets, labels) in enumerate(train_TB):
    if i > 5:
        break
    print(tweets.shape, labels.shape)


def get_accuracy(model, data_loader):
    correct, total = 0, 0  # ile ok, ile wszystkich
    for tweets, labels in data_loader:
        output = model(tweets)  # co mowi model
        pred = output.max(1, keepdim=True)[1]  # ktora kategoria
        correct += pred.eq(labels.view_as(pred)).sum().item()

        total += labels.shape[0]

    return correct / total


test_loader = TBatcher(test, batch_size=4, drop_last=False)
print(get_accuracy(model, test_loader))


def RNN_train(model, train, valid, num_epochs=5, learning_rate=1e-5):
    criterion = nn.CrossEntropyLoss()  # funkcja kosztu
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # optymalizator modelu
    losses, train_acc, valid_acc, epochs = [], [], [], []  # cztery listy na wartosci funkcji kosztu, dokladnosc na zbiorze testowym i walidacyjnym, numer epoki

    for epoch in range(num_epochs):  # przechodz kolejne epoki (iteracje)
        for tweets, labels in train:  # przechodzi dane ze zbioru testowego
            optimizer.zero_grad()  # zerowanie gradientu
            pred = model(tweets)  # co mowi model?
            loss = criterion(pred, labels)  # wartosc funkcji kosztu - porownanie tego co mowi model, a tego jak jest
            loss.backward()  # pochodna po funkcji kosztu
            optimizer.step()  # aktualizacja parametrow
        losses.append(float(loss))  # zapisz aktualną wartosc funkcji kosztu
        epochs.append(epoch)  # zapisz aktualny numer epoki
        train_acc.append(get_accuracy(model, train_loader))  # dokladnosc na zbiorze testowym
        valid_acc.append(get_accuracy(model, valid_loader))  # dokladnosc na zbiorze treningowym
        print(
            f'Epoch number: {epoch + 1} | Loss value: {loss} | Train accuracy: {round(train_acc[-1], 3)} | Valid accuracy: {round(valid_acc[-1], 3)}')

    # Rysowanie wykresow
    plt.title("Training Curve")
    plt.plot(losses, label="Train dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Loss value")
    plt.show()

    plt.title("Training Curve")
    plt.plot(epochs, train_acc, label="Train dataset")
    plt.plot(epochs, valid_acc, label="Validation dataset")
    plt.xlabel("Epoch number")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()


train_loader = TBatcher(train, batch_size=4, drop_last=True)  # dane treningowe z batchem
valid_loader = TBatcher(valid, batch_size=4, drop_last=False)

lstm_layer = nn.LSTM(input_size=200,  # wymiar wejscia
                     hidden_size=5,  # wymiar cech w stanie ukrytym
                     batch_first=True)

h0 = torch.zeros(1, 1, 5)  # początkowy stan ukryty
c0 = torch.zeros(1, 1, 5)  # początkowy cell state
out, last_hidden = lstm_layer(tweet_input, (h0, c0))


class T_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors)  # embeddingi
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM
        self.fc = nn.Linear(hidden_size, num_classes)  # przeksztalcenie liniowe

    def forward(self, x):
        x = self.emb(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size)  # początkowy  h0
        c0 = torch.zeros(1, x.size(0), self.hidden_size)  # początkowy c0
        out, _ = self.lstm(x, (h0, c0))  # LSTM
        out = self.fc(out[:, -1, :])  # przeksztlcam jeszcze liniowo ostatni output
        return out


model_lstm = T_LSTM(200, 50, 2)  # model
RNN_train(model_lstm, train_loader, valid_loader, num_epochs=17, learning_rate=0.00006)

print(get_accuracy(model_lstm, test_loader))
