# for consufion matrix
import pandas as pd
from sklearn.metrics import confusion_matrix

# already saved the data from the  algorithm NB.
df = pd.DataFrame(pd.read_csv('confusionMetrix_data.csv'))
# converting to numpu array
y_true = df['actual'].to_numpy()
y_pred = df['predicted'].to_numpy()
# printing the result using the confusion matrix of sklearn library
print("confusion matrix")
print(confusion_matrix(y_true, y_pred))

df2= pd.DataFrame(pd.read_csv('countTable.csv'))
txt=pd.DataFrame(pd.read_csv('vocabulary.txt'))
df2.drop('1',axis=1,inplace=True)
df2.drop('count',axis=1,inplace=True)
df3= pd.DataFrame(df2.sum(axis=0))
df3.drop(index='14',axis=0,inplace=True)
df3.drop(index='0',axis=0,inplace=True)
df3['vocublary']=txt.to_numpy()
maxWord=[]
for x in df3[0].nlargest(100):
    maxWord.append(df3.loc[df3[0]==x]['vocublary'][0])

print("Printing the 100 words appeared most frequently in the given datasets")
print(maxWord)

