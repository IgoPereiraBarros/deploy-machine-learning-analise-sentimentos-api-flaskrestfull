from flask import Flask
from flask_restful import reqparse, abort, Api, Resource

import pickle
import numpy as np

from model import NLPModel


app = Flask(__name__)
api = Api(app)

#create new model object
model = NLPModel()

clf_path = 'materials/models/SentimentClassifier.pkl'
with open(clf_path, 'rb') as f:
    model.clf = pickle.load(f)

vec_path = 'materials/models/TfidfVectorizer.pkl'
with open(vec_path, 'rb') as f:
    model.vectorizer = pickle.load(f)

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictSentiment(Resource):

    def get(self):
         # use parser and find the user's query
         args = parser.parse_args()
         user_query = args['query']

         # vectorize the user's query and make a prediction
         up_vectorizer = model.vectorizer_transform(np.array([user_query]))
         prediction = model.predict(up_vectorizer)
         pred_proba = model.predict_proba(up_vectorizer)

         # Output either 'Negative' or 'Positive' along with the score
         if prediction == 0:
            pred_text = 'Positive'
         else:
            pred_text = 'Negative'

         # round the predict proba value and set to new variable
         confidence = round(pred_proba[0], 2) * 100

         # create JSON object
         output = {
            'prediction': pred_text,
            'confidence': f'{confidence}%'
         }

         return output

# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictSentiment, '/')

if __name__ == '__main__':
    app.run(debug=True)


