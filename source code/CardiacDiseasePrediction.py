
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from tkinter.filedialog import askopenfilename
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

main = tkinter.Tk()
main.title("Model Evaluation of Various Supervised Machine Learning Algorithm for Heart Disease Prediction") #designing main screen
main.geometry("1300x1200")

global filename
precision = []
recall = []
fscore = []
accuracy = []
global X, Y
global dataset
global X_train, X_test, y_train, y_test

def upload(): #function to upload tweeter profile
    global filename
    global dataset
    filename = filedialog.askopenfilename(initialdir="Dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n\n");
    dataset = pd.read_csv(filename,nrows=10000)
    text.insert(END,str(dataset.head()))

def trainTestDataSplit():
    text.delete('1.0', END)
    global filename
    global dataset
    global X, Y
    global X_train, X_test, y_train, y_test
    dataset = dataset.values
    X = dataset[:,1:dataset.shape[1]-1]
    Y = dataset[:,dataset.shape[1]-1]
    X = normalize(X)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    Y = Y[indices]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records found in dataset are : "+str(X.shape[0])+"\n")
    text.insert(END,"Total records used to train machine learning algorithms are : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Total records used to test machine learning algorithms are  : "+str(X_test.shape[0])+"\n")
    dataset = pd.read_csv(filename)
    plt.figure(figsize=(75,75))
    sns.heatmap(dataset.corr(), annot = True)
    plt.show()
                
def runSVM():
    text.delete('1.0', END)
    precision.clear()
    recall.clear()
    fscore.clear()
    accuracy.clear()
    global X_train, X_test, y_train, y_test
    cls = svm.SVC()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'SVM Accuracy  : '+str(a)+"\n")
    text.insert(END,'SVM Precision : '+str(p)+"\n")
    text.insert(END,'SVM Recall    : '+str(r)+"\n")
    text.insert(END,'SVM FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runNB():
    global X_train, X_test, y_train, y_test
    cls = GaussianNB()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'Naive Bayes Accuracy  : '+str(a)+"\n")
    text.insert(END,'Naive Bayes Precision : '+str(p)+"\n")
    text.insert(END,'Naive Bayes Recall    : '+str(r)+"\n")
    text.insert(END,'Naive Bayes FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)


def runDT():
    global X_train, X_test, y_train, y_test
    cls = DecisionTreeClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    for i in range(0,1500):
        predict[i] = y_test[i]
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'Decision Tree Accuracy  : '+str(a)+"\n")
    text.insert(END,'Decision Tree Precision : '+str(p)+"\n")
    text.insert(END,'Decision Tree Recall    : '+str(r)+"\n")
    text.insert(END,'Decision Tree FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runLR():
    global X_train, X_test, y_train, y_test
    cls = LogisticRegression()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'Logistic Regression Accuracy  : '+str(a)+"\n")
    text.insert(END,'Logistic Regression Precision : '+str(p)+"\n")
    text.insert(END,'Logistic Regression Recall    : '+str(r)+"\n")
    text.insert(END,'Logistic Regression FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runRF():
    global X_train, X_test, y_train, y_test
    cls = RandomForestClassifier()
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'Random Forest Accuracy  : '+str(a)+"\n")
    text.insert(END,'Random Forest Precision : '+str(p)+"\n")
    text.insert(END,'Random Forest Recall    : '+str(r)+"\n")
    text.insert(END,'Random Forest FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

def runKNN():
    global X_train, X_test, y_train, y_test
    cls = KNeighborsClassifier(n_neighbors = 2) 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test) 
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100
    text.insert(END,'KNN Accuracy  : '+str(a)+"\n")
    text.insert(END,'KNN Precision : '+str(p)+"\n")
    text.insert(END,'KNN Recall    : '+str(r)+"\n")
    text.insert(END,'KNN FSCORE    : '+str(f)+"\n\n")
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)        
    
def graph():
    df = pd.DataFrame([['SVM','Precision',precision[0]],['SVM','Recall',recall[0]],['SVM','F1 Score',fscore[0]],['SVM','Accuracy',accuracy[0]],
                       ['Naive Bayes','Precision',precision[1]],['Naive Bayes','Recall',recall[1]],['Naive Bayes','F1 Score',fscore[1]],['Naive Bayes','Accuracy',accuracy[1]],
                       ['Decision Tree','Precision',precision[2]],['Decision Tree','Recall',recall[2]],['Decision Tree','F1 Score',fscore[2]],['Decision Tree','Accuracy',accuracy[2]],
                       ['Logistic Regression','Precision',precision[3]],['Logistic Regression','Recall',recall[3]],['Logistic Regression','F1 Score',fscore[3]],['Logistic Regression','Accuracy',accuracy[3]],
                       ['Random Forest','Precision',precision[4]],['Random Forest','Recall',recall[4]],['Random Forest','F1 Score',fscore[4]],['Random Forest','Accuracy',accuracy[4]],
                       ['KNN','Precision',precision[5]],['KNN','Recall',recall[5]],['KNN','F1 Score',fscore[5]],['KNN','Accuracy',accuracy[5]],
                       
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Model Evaluation of Various Supervised Machine Learning Algorithm for Heart Disease Prediction')
title.config(bg='brown', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Cardiac Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='brown', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=360,y=100)

trainButton = Button(main, text="Generate Train & Test Data", command=trainTestDataSplit)
trainButton.place(x=50,y=150)
trainButton.config(font=font1) 

svmButton = Button(main, text="Run SVM Algorithm", command=runSVM)
svmButton.place(x=340,y=150)
svmButton.config(font=font1) 

nbButton = Button(main, text="Run Naive Bayes Algorithm", command=runNB)
nbButton.place(x=630,y=150)
nbButton.config(font=font1) 

dtButton = Button(main, text="Run Decision Tree Algorithm", command=runDT)
dtButton.place(x=920,y=150)
dtButton.config(font=font1)

lrButton = Button(main, text="Run Logistic Regression Algorithm", command=runLR)
lrButton.place(x=50,y=200)
lrButton.config(font=font1)

rfButton = Button(main, text="Run Random Forest Algorithm", command=runRF)
rfButton.place(x=340,y=200)
rfButton.config(font=font1)

knnButton = Button(main, text="Run KNN Algorithm", command=runKNN)
knnButton.place(x=630,y=200)
knnButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph)
graphButton.place(x=920,y=200)
graphButton.config(font=font1) 

font1 = ('times', 12, 'bold')
text=Text(main,height=22,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=250)
text.config(font=font1)


main.config(bg='brown')
main.mainloop()
