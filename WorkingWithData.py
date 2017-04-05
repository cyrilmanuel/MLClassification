import sklearn.datasets
from array import array
import sklearn

# définition des catégories
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

# load train data for classification
twenty_train = sklearn.datasets.load_files("20news-bydate-train", description=None, categories=categories, load_content=True, shuffle=True, encoding='latin-1', decode_error='strict', random_state=42)

# création dictionnaire pour contenire les nom catégorie en interne
twenty_train.target_names
['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']

# teste longueur obtient 2257
print(len(twenty_train.data))
print(len(twenty_train.filenames))


# test en imprimant la première ligne du premiere fichier loader et sa catégorie :
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print(twenty_train.target_names[twenty_train.target[0]])


twenty_train.target[:10]

#2. vectorisation
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
