from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud
import string
from nltk.corpus import stopwords
import nltk
import re
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

resumeDataSet = pd.read_csv('./UpdatedResumeDataSet.csv', encoding='utf-8')
resumeDataSet['cleaned_resume'] = ''
resumeDataSet.head()

resumeDataSet.info()

print("Displaying the distinct categories of resume:\n\n ")
print(resumeDataSet['Category'].unique())

print("Displaying the distinct categories of resume and the number of records belonging to each category:\n\n")
print(resumeDataSet['Category'].value_counts())

plt.figure(figsize=(20, 5))
plt.xticks(rotation=90)
ax = sns.countplot(x="Category", data=resumeDataSet)
for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1.01, p.get_height() * 1.01))
plt.grid()

targetCounts = resumeDataSet['Category'].value_counts()
targetLabels = resumeDataSet['Category'].unique()
# Make square figures and axes
plt.figure(1, figsize=(22, 22))
the_grid = GridSpec(2, 2)


cmap = plt.get_cmap('coolwarm')
plt.subplot(the_grid[0, 1], aspect=1, title='CATEGORY DISTRIBUTION')

source_pie = plt.pie(targetCounts, labels=targetLabels,
                     autopct='%1.1f%%', shadow=True)
plt.show()


def cleanResume(resumeText):
    resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@\S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^\x00-\x7f]', r' ', resumeText)
    resumeText = re.sub('\s+', ' ', resumeText)  # remove extra whitespace
    return resumeText


resumeDataSet['cleaned_resume'] = resumeDataSet.Resume.apply(
    lambda x: cleanResume(x))

resumeDataSet.head()

resumeDataSet_d = resumeDataSet.copy()


oneSetOfStopWords = set(stopwords.words('english')+['``', "''"])
totalWords = []
Sentences = resumeDataSet['Resume'].values
cleanedSentences = ""
for records in Sentences:
    cleanedText = cleanResume(records)
    cleanedSentences += cleanedText
    requiredWords = nltk.word_tokenize(cleanedText)
    for word in requiredWords:
        if word not in oneSetOfStopWords and word not in string.punctuation:
            totalWords.append(word)

wordfreqdist = nltk.FreqDist(totalWords)
mostcommon = wordfreqdist.most_common(50)
print(mostcommon)

wc = WordCloud().generate(cleanedSentences)
plt.figure(figsize=(10, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()


var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    resumeDataSet[i] = le.fit_transform(resumeDataSet[i])

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

resumeDataSet.head()

resumeDataSet.Category.value_counts()

resumeDataSet_d.Category.value_counts()  # understanding decode LabelEncoder

del resumeDataSet_d  # clearing the space occupied


requiredText = resumeDataSet['cleaned_resume'].values
requiredTarget = resumeDataSet['Category'].values

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english')
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)

print("Feature completed .....")

X_train, X_test, y_train, y_test = train_test_split(WordFeatures, requiredTarget, random_state=42, test_size=0.2,
                                                    shuffle=True, stratify=requiredTarget)
print(X_train.shape)
print(X_test.shape)

clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print('Accuracy of KNeighbors Classifier on training set: {:.2f}'.format(
    clf.score(X_train, y_train)))
print('Accuracy of KNeighbors Classifier on test set:     {:.2f}'.format(
    clf.score(X_test, y_test)))

print("\n Classification report for classifier %s:\n%s\n" %
      (clf, metrics.classification_report(y_test, prediction)))

sample_resumes = {
    'Sample Resume 1': '''Computer Skills: Languages And Script: JSP, Servlet, HTML, CSS, Java Script, Jquery, Ajax, Spring, Hibernate Operating System: Microsoft WindowsÂ® 2003/XP/Vista/7/8 Databases: My SQL Concepts: OOPS, Core java, Advance java Development Tool: Eclipse, Net beans IDE Web Server: Apache Tomcat 7.0Education Details 
January 2007 H.S.C  Amravati, Maharashtra VidyaBharati college
January 2005 S.S.C  Amravati, Maharashtra Holy Cross English School
Java Developer 

Java Developer - Kunal IT Services Pvt Ltd
Skill Details 
ECLIPSE- Exprience - Less than 1 year months
JAVA- Exprience - 14 months
HIBERNATE- Exprience - Less than 1 year months
SPRING- Exprience - Less than 1 year months
jQuery- Exprience - Less than 1 year monthsCompany Details 
company - Kunal IT Services Pvt Ltd
description - Currently Working As Java Developer In Winsol Solution Pvt Ltd From 1 July 2017 To Till Date.

Experience Of  2 Yrs As A Java Developer In Kunal IT Services Pvt Ltd.''',

    'Sample Resume 2': '''Technical Skills Summary I have completed ""CORPORATE TRAINING in Manual and Automation Testing"" at Source-Code Technology, Pune. â Manual and Automation Testing â¢ SELENIUM IDE, TestNG, SELENIUM Grid, JENKINS, Apache POI. â¢ Good knowledge in SDLC. â¢ Excellent understanding of White Box Testing and Black Box Testing. â¢ Good knowledge in Functional Testing, Integration Testing and System Testing. â¢ Good Exposure in writing Functional and Integration Scenarios. â¢ Good understanding of writing of test cases including test case design technique. â¢ Good understanding Build and release. â¢ Good knowledge on Ad hoc and smoke testing. â¢ Excellent understanding of usability, reliability and exploratory testing. â¢ Excellent knowledge of Globalization and Compatibility Testing. â¢ Excellent Understand of STLC. â¢ Good knowledge of regression and retesting. â¢ Excellent knowledge on Defect tracking and Defect Life Cycle. â¢ Good Knowledge on Test plan and Traceability Matrix. Internship Project Project Name: Resume Extractor Duration: 6 months Role: Manual And Automation Testing Environment: Jdbc, Servlets, Jsp, Technologies: Web based Application, MS Access2007 The project involved development of a web application. Resume extractor provides the technology to analyze mass volume of data to detect resume in data covered which company have into valuable information. This project is company site's based on recruitment process. Strengths â¢ Able to work in a team â¢ System and Operational Analysis â¢ Good Communication Skills â¢ Active learning and critical thinking â¢ Good interpersonal skills, willing to take challenges and more responsibilities. â¢ Ability to learn new technologies with minimal time period. Education Details 
January 2015  BCS Computer Science  MGM's College
 MCA  Pune, Maharashtra Computer Science fromJSPM College
 HSC  Nanded, Maharashtra Maharashtra state board
 SSC  Nanded, Maharashtra Maharashtra State Board
Software testing 

Software testing
Skill Details 
APACHE- Exprience - 6 months
BLACK BOX- Exprience - 6 months
BLACK BOX TESTING- Exprience - 6 months
FUNCTIONAL TESTING- Exprience - 6 months
INTEGRATION- Exprience - 6 monthsCompany Details 
company - Tech Mahindra
description - Software testing in manual and Automation
company - 
description - software Test engineer''',

    'Sample Resume 3': '''"Software Proficiency: â¢ Languages: Basics of C, SQL, PL/SQL,JAVA,JAVAEE,Javascript,HTML,CSS,jquery,mysql,Spring ,Hibernate. â¢ Software Tools: Xillinx, Modelsim, Matlab, Multisim. â¢ Operating Systems: Windows XP, Vista, 07, 08, Ubuntu. Project Profile: B.E. Project FPGA Implementation of Team Size: 4. Role: Programmer. AES Algorithm AES is Advanced Encryption Standard which is used in cryptography by which we can protect our data. It encrypted by a Secret Key. T.E. project Sorting Robot. Team Size: 3. Role: Mechanism designer. The TCS 230 sensor sorts the RGB color balls according to their color. Diploma Project RFID Based Student Team Size: 4. Role: Interface. Attendance System Using GSM. In this student show RFID card of his own and then message send via GSM to their parent that his ward is present.Education Details 
May 2016 B.E. Savitribai Phule Pune, Maharashtra Pune University
March 2010 S.S.C   Maharashtra Board
DevOps Engineer 


Skill Details 
C- Exprience - 6 months
C++- Exprience - 6 months
Sql- Exprience - 6 months
Pl/Sql- Exprience - 6 months
Core Java- Exprience - 6 months
Javascript- Exprience - Less than 1 year months
HTML- Exprience - Less than 1 year months
CSS- Exprience - Less than 1 year months
Jquery- Exprience - Less than 1 year months
JavaEE- Exprience - Less than 1 year months
Mysql- Exprience - Less than 1 year months
Python- Exprience - 6 monthsCompany Details 
company - Parkar Consulting and Labs
description - I'm working on the DevOps team in Parkar Consulting and Labs. I have hands on the AWS as well as Python''',
}


def predict_resume(resume_text):
    # clean the text
    resume_text = cleanedText(resume_text)

    # extract features
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        stop_words='english'
    )

    X = vectorizer.fit_transform([resume_text])
    X = X.toarray()

    # predict
    return clf.predict(X)
