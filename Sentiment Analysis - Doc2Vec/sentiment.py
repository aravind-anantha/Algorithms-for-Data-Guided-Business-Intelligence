import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
#nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)

    with open('train_pos', 'wb') as fp:
        pickle.dump(train_pos_vec, fp)
    with open('train_neg', 'wb') as fp:
        pickle.dump(train_neg_vec, fp)
    with open('test_pos', 'wb') as fp:
        pickle.dump(test_pos_vec, fp)
    with open('test_neg', 'wb') as fp:
        pickle.dump(test_neg_vec, fp)

    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)
    return train_pos, train_neg, test_pos, test_neg



def generate_binary_vector(words_list, feature_words):
    res = [[0] * len(feature_words.keys()) for __ in range(len(words_list))]
    for index, sentence in enumerate(words_list):
        for word in sentence:
            if feature_words.has_key(word):
                res[index][feature_words[word]] = 1
    return res
def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    
    train_pos = [[w for w in sentence if w not in stopwords] for sentence in train_pos]
    train_neg = [[w for w in sentence if w not in stopwords] for sentence in train_neg]
    test_pos = [[w for w in sentence if w not in stopwords] for sentence in test_pos]
    test_neg = [[w for w in sentence if w not in stopwords] for sentence in test_neg]
    word_count = {}
    for sentence in train_pos:
        for word in set(sentence):
        	if word_count.has_key(word):
				word_count[word][0] += 1
        	else:
				word_count[word] = [1, 0]

    for sentence in train_neg:
        for word in set(sentence):
            if word_count.has_key(word):
                word_count[word][1] += 1
            else:
                word_count[word] = [0, 1]

    feature_words = {}
    len_pos = len(train_pos)
    len_neg = len(train_neg)
    value = 0
    for word, count in word_count.items():
        if count[0] >= len_pos * 0.01 or count[1] >= len_neg * 0.01:
            if count[0] >= count[1] * 2 or count[1] >= count[0] * 2:
                feature_words[word] = value
                value += 1

    #print feature_words
    #print len(feature_words)
    
    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    
    train_pos_vec = generate_binary_vector(train_pos, feature_words)
    train_neg_vec = generate_binary_vector(train_neg, feature_words)
    test_pos_vec = generate_binary_vector(test_pos, feature_words)
    test_neg_vec = generate_binary_vector(test_neg, feature_words)
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    
    labeled_train_pos = [LabeledSentence(words = sentence, tags = ['TRAIN_POS_' + str(i)]) for i, sentence in enumerate(train_pos)]
    labeled_train_neg = [LabeledSentence(words = sentence, tags = ['TRAIN_NEG_' + str(i)]) for i, sentence in enumerate(train_neg)]
    labeled_test_pos = [LabeledSentence(words = sentence, tags = ['TEST_POS_' + str(i)]) for i, sentence in enumerate(test_pos)]
    labeled_test_neg = [LabeledSentence(words = sentence, tags = ['TEST_NEG_' + str(i)]) for i, sentence in enumerate(test_neg)]
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)
    # Train the model
    # This may take a bit to run
    for i in range(15):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences, total_examples=model.corpus_count, epochs=15)
    # Use the docvecs function to extract the feature vectors for the training and test data
    
    model.save('doc2vec_big.model')
    train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = [], [], [], []
    for tag in model.docvecs.doctags.keys():
        if "TRAIN_POS_" in tag:
            train_pos_vec.append(model.docvecs[tag])
        elif "TRAIN_NEG_" in tag:
            train_neg_vec.append(model.docvecs[tag])
        elif "TEST_POS_" in tag:
            test_pos_vec.append(model.docvecs[tag])
        elif "TEST_NEG_" in tag:
            test_neg_vec.append(model.docvecs[tag])  
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    
    nb_model = BernoulliNB(alpha = 1.0, binarize = None).fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr_model = LogisticRegression().fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))

    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model = GaussianNB().fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    lr_model = LogisticRegression().fit(np.array(train_pos_vec + train_neg_vec), np.array(Y))
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    
    Y = np.array(["pos"]*len(test_pos_vec) + ["neg"]*len(test_neg_vec))
    test_data = np.array(test_pos_vec + test_neg_vec)
    pred = model.predict(test_data)
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(len(Y)):
        if Y[i] == pred[i] == 'pos':
            tp += 1
        if Y[i] == 'neg' and pred[i] == 'pos':
            fp += 1
        if Y[i] == pred[i] == 'neg':
            tn += 1
        if Y[i] == 'pos' and pred[i] == 'neg':
            fn += 1
    accuracy = float(tp + tn) / float(tp + fp + fn + tn)
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
