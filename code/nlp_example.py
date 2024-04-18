# import libraries
# sklearn reference: https://scikit-learn.org/
# pandas reference: https://pandas.pydata.org/
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import metrics
import pandas as pd

# text and numeric classes that use sklearn base libaries
class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Transform text features
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None, *parg, **kwarg):
        return self

    def transform(self, X):
        return X[self.key]
    
class NumberTransformer(BaseEstimator, TransformerMixin):
    """
    Transform numeric features
    """
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]

# read in your dataframe
df = pd.read_csv('/Users/data.csv')

# use the term-frequency inverse document frequency vectorizer to transfrom count of text
# into a weighed matrix of term importance
vec_tdidf = TfidfVectorizer(ngram_range=(1,1), analyzer='word', norm='l2')

# compile both the TextTransformer and TfidfVectorizer 
# to the text 'Text_Feature' 
color_text = Pipeline([
                ('transformer', TextTransformer(key='Text_Feature')),
                ('vectorizer', vec_tdidf)
                ])

# compile the NumberTransformer to 'Confirmed_Test', 'Confirmed_Recovery', 
# and 'Confirmed_New' numeric features
test_numeric = Pipeline([
                ('transformer', NumberTransformer(key='Confirmed_Test')),
                ])
recovery_numeric = Pipeline([
                ('transformer', NumberTransformer(key='Confirmed_Recovery')),
                ])
new_numeric = Pipeline([
                ('transformer', NumberTransformer(key='Confirmed_New')),
                ])

# combine all of the features, text and numeric together
features = FeatureUnion([('Text_Feature', color_text),
                      ('Confirmed_Test', test_numeric),
                      ('Confirmed_Recovery', recovery_numeric),
                      ('Confirmed_New', new_numeric)
                      ])

# create the classfier from RF
clf = RandomForestClassifier()

# unite the features and classfier together
pipe = Pipeline([('features', features),
                 ('clf',clf)
                 ])

# transform the categorical predictor into numeric
predicted_dummies = pd.get_dummies(df['Text_Predictor'])

# split the data into train and test
# isolate the features from the predicted field
text_numeric_features = ['Text_Feature', 'Confirmed_Test', 'Confirmed_Recovery', 'Confirmed_New']
predictor = 'Text_Predictor'

X_train, X_test, y_train, y_test = train_test_split(df[text_numeric_features], df[predictor], 
                                                    test_size=0.25, random_state=42)

# fit the model
pipe.fit(X_train, y_train)

# predict from the test set
preds = pipe.predict(X_test)

# print out your accuracy
print("Accuracy:",metrics.accuracy_score(y_test, preds))
