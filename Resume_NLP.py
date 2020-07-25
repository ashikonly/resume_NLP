# Module to convert pdf to text
from pdfminer.high_level import extract_text  # Package to read a pdf file to text

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import string
from matplotlib.gridspec import GridSpec
import pickle

import nltk  # Module for text processing
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from wordcloud import WordCloud  # Plotting the word_cloud

from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Creating select button to browse to the file paths of resume and jd
resume_location = str(input("Enter the Path for Resume:"))
jd_location = input("Enter the Jd:")
input_resume = extract_text(resume_location)
input_jd = str(jd_location)


# print(input_jd)
# print(input_resume)

# Regular expression function to clean the text
def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ',
                        resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


# importing the necessary package of the nltk for text processing
nltk.download('stopwords')
nltk.download('punkt')

# Cleaning the text in the resume and also displaying most common words as a word cloud
oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
totalWords_resume = []
Sentences_resume = input_resume
cleanedSentences_resume = ""
cleanedText_resume = cleanResume(input_resume)
cleanedText_resume = cleanedText_resume.lower()
cleanedSentences_resume += cleanedText_resume
requiredWords_resume = nltk.word_tokenize(cleanedText_resume)
porter = PorterStemmer()
for word in requiredWords_resume:
    if word not in oneSetOfStopWords and word not in string.punctuation:
        totalWords_resume.append(porter.stem(word))

input_resume_cleaned = ' '.join(totalWords_resume)
wordfreqdist_resume = nltk.FreqDist(totalWords_resume)
mostcommon_resume = wordfreqdist_resume.most_common(50)

print(mostcommon_resume)
print(cleanedText_resume)

wc_resume = WordCloud().generate(cleanedText_resume)
plt.figure(figsize=(15, 15))
plt.imshow(wc_resume, interpolation='bilinear')
plt.axis("off")
plt.show()

# Repeating the procedure for Jd
oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
totalWords_jd = []
Sentences_jd = input_jd
cleanedSentences_jd = ""
cleanedText_jd = cleanResume(input_jd)
cleanedText_jd = cleanedText_jd.lower()
cleanedSentences_jd += cleanedText_jd
requiredWords_jd = nltk.word_tokenize(cleanedText_jd)
for word in requiredWords_jd:
    if word not in oneSetOfStopWords and word not in string.punctuation:
        totalWords_jd.append(porter.stem(word))
wordfreqdist_jd = nltk.FreqDist(totalWords_jd)
mostcommon_jd = wordfreqdist_jd.most_common(50)
print(mostcommon_jd)
print(totalWords_jd)

wc_jd = WordCloud().generate(cleanedText_jd)
plt.figure(figsize=(15, 15))
plt.imshow(wc_jd, interpolation='bilinear')
plt.axis("off")
plt.show()

# Converting the distinct values in resume to a set
input_resume_w = set(totalWords_resume)
len(input_resume_w)
input_jd_w = totalWords_jd
len(input_jd_w)

# Calculating the score of how many distinct words in jd are present in resume
score = 0
for word_1 in input_jd_w:
    for word_2 in input_resume_w:
        if word_1 == word_2:
            score = score + 1
print(score)
Final_score_p = (score / len(input_jd_w)) * 100
print(Final_score_p)

# Initial steps in building a model for resume classification

warnings.filterwarnings('ignore')
resumeDataSet = pd.read_csv('E:\\Resume Project\\datasets_118409_284260_resume_dataset.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()
print("Displaying the distinct categories of resume -")
print(resumeDataSet['Category'].unique())
print("Displaying the distinct categories of resume and the number of records belonging to each category -")
print(resumeDataSet['Category'].value_counts())

plt.figure(figsize=(15, 15))
plt.xticks(rotation=90)
sns.countplot(y="Category", data=resumeDataSet)

# plotting the resume category in a pie chart
targetCounts = resumeDataSet['Category'].value_counts()
targetLabels = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(50, 50))
the_grid = GridSpec(2, 2)
cmap = plt.get_cmap('coolwarm')
colors = [cmap(i) for i in np.linspace(0, 1, 3)]
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels, autopct='%1.1f%%', shadow=True, colors=colors)
plt.show()

print(resumeDataSet.shape[0])

# Performing the entire prepossessing steps to build the model

oneSetOfStopWords = set(stopwords.words('english') + ['``', "''"])
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for i in range(0, resumeDataSet.shape[0]):
    totalWords = []
    cleanedText = cleanResume(Sentences[i])
    cleanedText = cleanedText.lower()
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(porter.stem(word))
    resumeDataSet['cleaned_resume'][i] = ' '.join(totalWords)

# resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(lambda x: cleanResume(x))
# print (resumeDataSet['cleaned_resume'][31])

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)

print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()

# Label encoding of the Resume Category in order to fit in to a model
var_mod = ['Category']
le = LabelEncoder()
le.fit(resumeDataSet[var_mod])
list(le.classes_)
le.transform(resumeDataSet[var_mod])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
resumeDataSet['Category'] = le.fit_transform(resumeDataSet['Category'])
print("CONVERTED THE CATEGORICAL VARIABLES INTO NUMERICALS")
print(le_name_mapping)

filename3 = 'le.pkl'
pickle.dump(le, open(filename3, 'wb'))


resumeDataSet.head()

requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

# Vectorizing to convert it to a sparse matrix for model building
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)

filename2 = 'word_vec.pkl'
pickle.dump(word_vectorizer, open(filename2, 'wb'))

WordFeatures = word_vectorizer.transform(requiredText)

print("Feature completed .....")

# Splitting the data to training and test set
X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=0, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

# Running  K Neighnour classifier on the test set and evaluating it with training set
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

# Running  MultinomialNB on the test set and evaluating it with training set
clf = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of MultinomialNB Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of MultinomialNB Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))

# Creating a data frame in the format for prediction
rez = pd.DataFrame({'Category_input': '', 'Resume_input': input_resume, 'cleaned_resume_input': input_resume_cleaned},
                   index=[0])
requiredText_new = rez['cleaned_resume_input'].values

# WordFeatures = word_vectorizer.transform(input_resume_cleaned)
WordFeatures_new = word_vectorizer.transform(requiredText_new)
prediction_final = clf.predict(WordFeatures_new)

print(le.inverse_transform(prediction_final))

# Creating a pickle file for the classifier
filename = 'resume_prediction.pkl'
pickle.dump(clf, open(filename, 'wb'))


