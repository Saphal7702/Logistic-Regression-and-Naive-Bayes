import math
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix

# Here we are importing data in csv format.
# countTable is compressed table form of the given raw data
data = pd.read_csv('countTable.csv')
data_test = pd.read_csv('testing.csv')

# Creating dataframe for training and testing data
df = pd.DataFrame(data=data)
df_test = pd.DataFrame(data=data_test)

# Here we drop ID for training data and count the number of Xi in Yk
df.drop('1',axis=1,inplace=True)
df2 = df.copy()
df2.drop('count',axis=1,inplace=True)
df['totalcount']=df2.sum(axis=1)

# Here we dropped IDs to create csr matrix
df_test.drop('12001',axis=1,inplace=True)
df.drop('14',axis=1,inplace=True)

# Converting into csr sparse matrix
df_matrix=csr_matrix(df.values)
df_test_matrix=csr_matrix(df_test.values)

# PY holds the value of P(Yk) for 20 classes
PY =[];
beta = 1/61188;

# MLE creates the log2(P(Yk)) for each classes and stores in array
def MLE(y):
  PY.append(math.log2(y/12000));

for x in range(0, 20):
  MLE(df_matrix[x,61188])

# Here we created a MAP function that compute the log2(P(Xi | Yk)) and stores in a matrix
def map_function():
  map_matrix = lil_matrix((20, 61188))
  for row in range(0,20):
    for col in range( 0,61188):
      count = df_matrix[row, col]
      totalword = df_matrix[row, 61189]
      value=((count + beta)/(totalword + 1))
      map_matrix[row,col] = math.log2(value)
  return map_matrix

# The size of the matrix is 20 x 61188
map_matrix = map_function()

# Here we transpose the map_matrix inorder to make prediction.
transposed_matrix = map_matrix.transpose()

# We take the dot product of the testing sparse matrix and MAP matrix
# It is not the final prediction.
predict = df_test_matrix.dot(transposed_matrix).toarray()

# This array holds all the predicted class of the testing data.
result = []

# Here we add each value of the MLE to the initial prediction.
# The highest probability gives the classification of the class.
for i in range(len(predict)):
  args = []
  for j in range(len(predict[i])):
    args.append(PY[j]+predict[i][j])
  result.append(args.index(max(args))+1)

# Creating CSV to submit in Kaggle
ID = []
for x in range(12002, 18775):
  ID.append(x)
df3 = pd.DataFrame(data=ID, columns={'id'})
df3['class']=result
df3.to_csv('prediction.csv',index=False)
