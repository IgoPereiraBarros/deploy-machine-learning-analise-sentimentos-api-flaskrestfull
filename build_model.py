import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from model import NLPModel


def build_model():
    model = NLPModel()

    # filename = os.path.join(
    #     os.path.dirname(__file__), 'chalicelib', 'all/train.tsv')
    path = 'materials/datasets/train.tsv'
    data = pd.read_csv(path, sep='\t')

    pos_neg = data[(data['Sentiment'] == 0) | (data['Sentiment'] == 4)]

    pos_neg['Binary'] = pos_neg.apply(lambda x: 0 if x['Sentiment'] == 0 else 1, axis=1)

    model.vectorizer_fit(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(pos_neg.loc[:, 'Phrase'])
    print('Vectorizer transform complete')
    y = pos_neg.loc[:, 'Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    model.plot_roc(X_test, y_test, 12, 12)

if __name__ == '__main__':
    build_model()