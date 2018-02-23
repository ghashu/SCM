#Rating Prediction using SVM-RBF
import pymysql
import time
import pandas as pd
from sqlalchemy import create_engine
from mysql import connector
from sklearn import preprocessing
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.model_selection import StratifiedKFold
from imblearn.combine import SMOTEENN
from sklearn.metrics import matthews_corrcoef
from sklearn.decomposition import PCA

def classifySCM():
    datasetFile=pd.read_csv("InputTrain.dat", sep=',',header=None)
    datasetFile.columns=['Rate','TfIdf','BigramProb','F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12']
    df = pd.DataFrame(datasetFile)
    datasetArray=df.values
    #Features and Label
    X=datasetArray[:,1:]
    y=datasetArray.T[0,:]
    #smt = SMOTEENN(random_state=42)
    #X, y = smt.fit_sample(X, y)
    # Dimensionality Reduction on Features
    pca = PCA().fit(X)
    X = pca.fit_transform(X)
    #Split into dataset (default=0.25) for Training and Validation
    X_train, X_validation, y_train, y_validation = train_test_split(X, y,test_size=0.25 ,random_state=42,stratify=y)
    #X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=42)
    #Pre-processing Features of Training dataset
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train_transformed = scaler.fit_transform(X_train)
    X_validation_transformed = scaler.fit_transform(X_validation)
    #Hypothesis parameters
    parameter_candidates=[
                            {'C': [1,10,100,1000],
                             'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
                             'kernel': ['rbf'],
                             'class_weight': ['balanced'],
                             'decision_function_shape': ['ovr'],
                             'random_state':[0]
                            }
                         ]
    # Stratified Shuffle techniques
    cvval=StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
    #Model creation
    clf = GridSearchCV(estimator=SVC(), param_grid=parameter_candidates,cv=cvval)
    expected=y_validation
    #Model fitting
    clf.fit(X_train_transformed,y_train)
    clf.score(X_validation_transformed, y_validation)
    #Results
    print('Best score:', clf.best_score_)
    print('Best Parameters:', clf.best_params_)
    #Prediction
    predicted = clf.predict(X_validation_transformed)
    print("MCC:", matthews_corrcoef(expected, predicted))
    zarray=np.array([expected,predicted])
    zarray=zarray.T
    print(metrics.classification_report(expected, predicted))
    with open("ValResult.dat","wb") as f:
        np.savetxt(f,zarray,fmt=['%2.2f','%2.2f'])
    #Testing the model on Test dataset
    datasetTestFile = pd.read_csv("InputTest.dat", sep=',', header=None)
    datasetTestFile.columns = ['Engname','Comp','TfIdf','BigramProb','F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7',
                               'F8', 'F9', 'F10','F11', 'F12']
    dfTest = pd.DataFrame(datasetTestFile)
    testDatasetArray = dfTest.values
    # Features and Label
    X_test = testDatasetArray[:, 2:]
    # Dimensionality Reduction on Features
    pca = PCA().fit(X_test)
    X_test = pca.fit_transform(X_test)
    # Pre-processing Features of Test dataset
    scaler = preprocessing.StandardScaler()
    scaler.fit(X_test)
    X_test_transformed = scaler.fit_transform(X_test)
    #Predict
    predicted_test = clf.predict(X_test_transformed)
    print(predicted_test)
    dfTest['PredictedRating']=predicted_test
    dfTest.to_csv("SVMTestResult.dat",sep=',',header=None)
    #Load data to database table-svmengratingnogcd
    conn = create_engine('mysql+mysqlconnector://root:scm@localhost:3306/scm', echo=False)
   
    conn.execute("use scm;")
    conn.execute("SET SQL_SAFE_UPDATES=0;")
    conn.execute("delete from svmengratingng;")
    conn.execute('ALTER TABLE svmengratingng AUTO_INCREMENT = 1;')
    dfTest.columns=['Engname','Comp','TfIdf','NgramProb','Feature1', 'Feature2', 'Feature3', 'Feature4', 'Feature5',
                    'Feature6', 'Feature7','Feature8', 'Feature9', 'Feature10','Feature11', 'Feature12','PredRating']
    dfTest.to_sql(con=conn, name='svmengratingng',if_exists='append',index=False)

def loadTrainData():
    #Connect to database
    conn = pymysql.connect(host='localhost', user='root', passwd='root', db='scm')
    cursor = conn.cursor()
    #Execute SQL to store Training Data-A.TfIdf,A.BigramProb,
    cursor.execute(
        "select A.ActualLevel,A.TfIdf,A.NgramProb,B.Feature1,B.Feature2,B.Feature3,B.Feature4,B.Feature5,B.Feature6,"
        "B.Feature7,B.Feature8,B.Feature9,B.Feature10,B.Feature11,B.Feature12 "
        "from grptcommontfidf A inner join gcmnfeatureweight B "
        "on A.Engname=B.Engname and A.Comp=B.Comp  where "
        "B.Feature1 <> 0.00 or B.Feature2 <> 0.00 or B.Feature3 <> 0.00 or B.Feature4 <> 0.00 or B.Feature5 <> 0.00 or "
        "B.Feature6 <> 0.00 or B.Feature7 <> 0.00 or B.Feature8 <> 0.00 or B.Feature9 <> 0.00 or B.Feature10 <> 0.00 or "
        "B.Feature11 <> 0.00 or B.Feature12 <> 0.00 order by A.ActualLevel desc;")
    results = cursor.fetchall()
    #Write Training Data into InputTrain.dat file
    with open("InputTrain.dat", "w") as writefile:
        for row in results:
            writefile.write("%s\n" % str(row))
    #Replace unnecessory characters like (,)
    with open('InputTrain.dat', 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('(', '')
    filedata = filedata.replace(')', '')
    with open('InputTrain.dat', 'w') as file:
        file.write(filedata)
    conn.close()

def loadTestData():
    conn = pymysql.connect(host='localhost', user='root', passwd='root', db='scm')
    cursor = conn.cursor()
    cursor.execute(
        "select A.Engname,A.Comp,A.TfIdf,A.NgramProb,B.Feature1,B.Feature2,B.Feature3,B.Feature4,B.Feature5,"
        "B.Feature6,B.Feature7,B.Feature8,B.Feature9,B.Feature10,B.Feature11,B.Feature12 "
        "from comptfidf A inner join gnotcmnfeatureweight B "
        "A.TfIdf <> 0.00 or A.NgramProb <> 0.00 or "
        "B.Feature1 <> 0.00 or B.Feature2 <> 0.00 or B.Feature3 <> 0.00 or B.Feature4 <> 0.00 or B.Feature5 <> 0.00 or "
        "B.Feature6 <> 0.00 or B.Feature7 <> 0.00 or B.Feature8 <> 0.00 or B.Feature9 <> 0.00 or B.Feature10 <> 0.00 or "
        "B.Feature11 <> 0.00 or B.Feature12 <> 0.00;")
    results = cursor.fetchall()
    with open("InputTest.dat", "w") as writefile:
        for row in results:
            writefile.write("%s\n" % str(row))
            # Replace unnecessory characters like (,)
    with open('InputTest.dat', 'r') as file:
        filedata = file.read()
    filedata = filedata.replace('(\'','')
    filedata = filedata.replace('\'','')
    filedata = filedata.replace('0)', '0')
    with open('InputTest.dat', 'w') as file:
        file.write(filedata)
    conn.close()

def loadPredictedRating():
    conn = pymysql.connect(host='localhost', user='root', passwd='root', db='scm')
    cursor = conn.cursor()
    cursor.execute("use scm;")
    cursor.execute("SET SQL_SAFE_UPDATES=0;")
    cursor.execute("delete from svmcomparerating;")
    cursor.execute(
        "insert into svmcomparerating(select A.RowNo,A.Engname,A.competence,B.Rating,A.predrating "
        "from svmengratingng A join grptnotcommoneng B on A.Engname=B.Engname and trim(A.Comp)= trim(B.Comp));"
    )
    conn.commit()
    conn.close()

if __name__ == '__main__':
    loadTrainData()
    #time.sleep(10)
    print("Training Data Loaded")
    loadTestData()
    #time.sleep(10)
    print("Test data Loaded")
    classifySCM()
    print("Classifier Completed")
    loadPredictedRating()
    print("Predicted Rating Loaded")
