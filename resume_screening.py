# Importing the required libraries
import pandas as pd
import string
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

nltk.download('wordnet')
nltk.download('stopwords')

# Loading the csv file data
df = pd.read_csv('data/ResumeDataset.csv')

# Displaying the top 5 rows of data
print("\n Top 5 rows of data : \n", df.head())

# Displaying the information about data
print("\n Information about data : \n")
df.info()

# Displaying the shape of data
print("\n Shape of data : \n", df.shape)

# Checking for null values
print("\n Null Values : \n", df.isnull().sum())

# EDA
# Finding out the counts of each category
print("\n Category Counts : \n")
print(df['Category'].value_counts())

# Count plot for counts of each category
sns.countplot(y='Category', data=df)
plt.title('Value Counts for Category Column')
plt.show()


# Data Cleaning
# Removing URLS
def remove_urls(text):
    text = re.sub(r"(https://[a-z0-9_.\-]+\.[a-z0-9_\-/]+)", "", text)
    text = re.sub(r"(http://[a-z0-9_.\-]+\.[a-z0-9_\-/]+)", "", text)
    text = re.sub(r"(www\.[a-z0-9_.\-]+\.[a-z0-9_\-/]+)", "", text)
    return text


df['Cleaned_Resume'] = df['Resume'].apply(remove_urls)

# Removing punctuations, accented characters, tokenizing and lemmatizing
lm = WordNetLemmatizer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]  # cloning the list using list slicing
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    for i in text:
        y.append(lm.lemmatize(i))

    return " ".join(y)


df['Cleaned_Resume'] = df['Cleaned_Resume'].apply(transform_text)

# Encoding the categorical into numeric
tfidf = TfidfVectorizer()
lb = LabelEncoder()

# Splitting the data into x and y
x = tfidf.fit_transform(df['Cleaned_Resume'])
y = lb.fit_transform(df['Category'])

# Splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Fitting the model
model = MultinomialNB()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)
y1_pred = model.predict(x_train)

# Accuracy and F1 Score
print("Train Accuracy : \n", accuracy_score(y_train, y1_pred))
print("Test Accuracy : \n", accuracy_score(y_test, y_pred))

print("F1 Score : \n", f1_score(y_test, y_pred, average='weighted'))

