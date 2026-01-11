import pandas as pd
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english')]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin1')
df = df[['v1','v2']]
df.columns = ['target','text']
df['target'] = df['target'].map({'ham':0,'spam':1})

# Transform text
df['transformed_text'] = df['text'].apply(transform_text)

# Vectorization
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text'])
y = df['target']

# Train model ✅
model = MultinomialNB()
model.fit(X, y)

# Save vectorizer & model ✅
pickle.dump(tfidf, open("vectorizer.pkl","wb"))
pickle.dump(model, open("model.pkl","wb"))

print("✅ Model and vectorizer saved successfully")
