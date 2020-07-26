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

        resumeDataSet.head()

        requiredText = resumeDataSet['cleaned_resume'].values
        requiredTarget = resumeDataSet['Category'].values

        # Vectorizing to convert it to a sparse matrix for model building
        word_vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            stop_words='english',
            max_features=1500)
        word_vectorizer.fit(requiredText)
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

        print("\n Classification report for classifier %s:\n%s\n" % (
        clf, metrics.classification_report(y_test, prediction)))
        # print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, prediction))

        # Running  MultinomialNB on the test set and evaluating it with training set
        clf = OneVsRestClassifier(MultinomialNB()).fit(X_train, y_train)
        prediction = clf.predict(X_test)
        print('Accuracy of MultinomialNB Classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
        print('Accuracy of MultinomialNB Classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))
        print("\n Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, prediction)))
