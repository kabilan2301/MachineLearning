import sklearn
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer

positive_set = [
  "We are Loved", "We are Lucky", "We are Good", "We are Best", "We wil Win"
]
negative_set = [
  "We are Hated", "We are Unlucky", "We are bad", "We are Worst", "We wil Lose"
]
sample_set = [
  "We are Lucky", "We are Good", "We wil Win", "We are Unlucky", "We are bad",
  "We wil Lose"
]
data_set = positive_set + negative_set
data_labels = ['POSITIVE'] * len(positive_set) + ['NEGATIVE'] * len(negative_set)
#print(data_set)
#print(data_labels)
vectorizer = CountVectorizer()
vectorizer.fit(data_set)
data_vectors = vectorizer.transform(data_set)
sample_vectors = vectorizer.transform(sample_set)
feature_names = vectorizer.get_feature_names_out()
model = tree.DecisionTreeClassifier()
model.fit(data_vectors, data_labels)
prediction = model.predict(sample_vectors)
print(sample_set)
print(prediction)