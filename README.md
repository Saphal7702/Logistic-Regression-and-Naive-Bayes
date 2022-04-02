# Project Classifier.

In this project, a large dataset of approximately 20,000 newsgroup entries is used and the main objective of the program is to correctly classify the given document belongs to a certain newsgroup. Since, it is a classification problem. Two popular and effective classification algorithm are used for this machine learning project: Naive Bayes classifier and Logistic Regression Classifier. Detail description of the working mechanism of program and the methods used is described below. 

## To run the program: 
    you specially need 64 bit compiler we are using python 3.10 64 bit version compiler.There will be a photo in case you wanted to kow the configuration of the compiler.Also the compiler is provided in the project submission but libraries are not provided. you need to download scipy,numpy and pandas in your project and configure that in your project interpreter. Also, due to the large training and testing data, we have not included in the submission file. In order to run the code, please keep both of the files in the same directory. 
    
## NaiveBayes.py
    Python program which perform naive bayes algorithm. First the model is trained with the training data which was provided, and later the model uses testing data to accurately predict newsgroup. First and foremost, the data is imported from the dataset in csv format into the dataframe. Also we create a count table which is a compressed version of the table from the given data. Likewise, we create training and testing dataframe from the provided dataset. After this value of random variable Xi in the i position is counted inorder to predit the Y value of the document. After this the dataset is converted into sparse matrix. 

## PreProcesingData:
    we have done preprossing of data code for that is in seperate python file preprocessingData.py.There you can pring the countTable.csv file

## Confusion matrix and 100 Highest word method
    confusion matrix and 100 highest word method are implented in separate file confusion&100highestword.py file. we imported the file with actual and predicted vlaue from trainign vlaue.
    Also we have used the pre processed data to pring the 100 highest count word.

### Method

**MLE(y):**
    Method that creates the log2(p(yk)) for 20 clasees and stores in the array. The formula is implemented accordingly to the documentation provided in the project description.

**map_function():**
    Method which uses the formula to calculate P(X|Y) using Map estimate. Here the method computes log2(P(Xi| Yk)) and stores it in a matrix

    After the model is trained, we then begin to test the model with the testing dataset. FIrst, the matrix is transposed and then the dot product of the testing sparse matrix and Map matrix is taken. All the predicted class is stored in the array. Later we add the MLE to the initial prediction and the class with highest probability gives the classification of that particular task. Later a csv file is created for the submission of the prediction online. 

## logisticRegression.py
    Python program which perform logistic regression algorithm for the classification problem given in the description. First the model is trained with the training data which was provided, and later the model uses testing data to accurately predict newsgroup. Here in this code, we did use scipy library of python. We tried to implement softmax function from our own, but it took alot of time and the implementation was not effective so we took an alternate solution by implementing softmax function found in the scipy library which made our program effective. Also, in the program you will find the commented code for our own function of softmax, just because of its ineffectiveness we didnt use that function. 

### Method

**y_matrix_encode(classes):**
    Method which creates matrix of 61188 * 20 dimension for the multiplication of all the values that are 0 except the given classifier value is one.
    returns: calculated matrix.

**lastPartofFormula(X,Y,W,Lamda):**
    Method which calculates the last part of the formula, where it updates the w value. 
    returns: the last term

**updateW(X,Y,atea,lamda):**
    Method which returns the updated value of W. Here, first a matrix w is initialized with the given row and column. Also matrix Y is also initialized. To minimize the error, the range is set to 1000, with every iteration the value of W is updated.
    you could update the W value in a for loop for 1000 iteration or more to get a good result.returns: updated value of W.

**classifier(feature):**
    Method used to predict the value of the classifier. Here the value read from the training data is passed. This method updates the value, calculated the predicted value of each instances and then saves the result to a csv file. 




