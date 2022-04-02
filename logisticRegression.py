# sources used: stackoverflow, Google, wikepidia.
import numpy as np
import pandas as pd
from scipy.sparse import  lil_matrix
from scipy.special import softmax

# implementation of softmax function i.e e^yi/sigma(e^y1) from j=1 to n
#But we find more convinent to use scipy softmax as it is also able to calculate the exponent of larger
#value our function gives nan for exponent of larger value.
# def softmax(x):
#     np_array_x=np.exp(x.toarray())
#     max= np.max(np_array_x,axis=1,keepdims=True)
#     e_x = np.exp(np_array_x-max)
#     sum = np.sum(e_x, axis=1, keepdims=True)
#     f_x = e_x / sum
#     return f_x

# creates a matrix of the 61188 colum and 20 row to do a multiplication
#  and all the value  are 0 except the given classifier value is one.
def y_matrix_encode(classes):
    # defined columns
    lisofColum = []
    for x in range(1, 21):
        lisofColum.append(str(x))
    # creted a dataFrame from numpy zeros.
    data = np.zeros([11999, 20])
    df = pd.DataFrame(data, columns=lisofColum)
    # matrix creation
    df_matr = lil_matrix(df.values)
    # iterating in dataframe to get the  new matrix
    row = 0
    for x in classes['14']:
        df_matr[row, x-1] = 1
        row += 1

    # returning the matrix.
    return df_matr

def lastPartofFormula(X, Y, W, lamda):
    const1 = X @ W
    # taking softmax
    softmax_var= softmax(const1.toarray(),axis=1)
    X_T=X.transpose()
    # formula to update w
    #X_T =is the transpose of X matrix for multiplicaiton.
    # delat is Y comming from Y_matrix_encode function()

    last_term = (X_T @ (Y-softmax_var)) - lamda * W
    return last_term
def updateW(X, atea=0.01, lamda=0.001):
    # initializing the W to be of row= 61188 and colum= 20.
    W= lil_matrix(np.zeros([61188,20]))
    # creating a  matrix of Y whose shape is equal to row 61188 and colum is 20 for multiplication.
    ecoded_class_matrix=y_matrix_encode(Y_df)
    # can wrap the formula in for loop in order to minize the error.
    # updating the walue of W  by multipling it with eta
    # uncomment and indent for loop with W value to get good result.
    # for x in range(0,1000):
    W += atea * lastPartofFormula(X, ecoded_class_matrix, W, lamda)
    # returning the vlaue of W from this function.
    return  W

#  method to predict the result value classifier.
def classifier(feature):
    const2 =  feature @ W
    # result data class array
    class_data=(np.argmax(softmax(const2.toarray(),axis=1), axis=1)+1)
    # id class value
    ID = []
    for x in range(12002, 18775):
        ID.append(x)
    df3 = pd.DataFrame(data=ID, columns={'id'})
    df3['class']=class_data
    # saving the result class of finalResult.csv file
    df3.to_csv("finalPre.csv",index=False)
    return(df3)


# reading the training.csv file using pandas.read_csv
x_train=pd.read_csv('training.csv')
# creating a dataFrame
X_df = pd.DataFrame(data=x_train)
# Droping of the id colum
X_df.drop('1',axis=1,inplace=True)
# training vlaue assigning to a Y_df variable
Y_df =pd.DataFrame(X_df['14'])
# creating a sparse matrix of type lil_matrix
Y=lil_matrix(Y_df.values)
#  reshaping Y matrix for multiplication.
Y.reshape(11999,1)
# droping the prediction colum from the feature as it is assign to another matrix
X_df.drop('14',axis=1,inplace=True)
# creating the sparse matrix of row 1199 and colum 61188.
X=lil_matrix(X_df.values)
#  reading the testing file.
data2=pd.read_csv('testing.csv')
# converting trainign file to a dataframe
df_testing= pd.DataFrame(data2)
#  dropping index colum from dataframe.
df_testing.drop('12001',axis=1,inplace=True)
# converting into a sparse matrix of  type lil_matrix as it doesnot store 0.
x_preicate_matrix=lil_matrix(df_testing.values)
# Updated value of W from our model as it is assign to zero vlaues from above method.
W = updateW(X)
# classifing the classes using the classifier function.
result=classifier(x_preicate_matrix)

print("printing the resulting class of 6673 rows as numpy array")
print(result)
