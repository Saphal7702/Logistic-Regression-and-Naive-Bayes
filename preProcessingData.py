# sources used: stackoverflow, Google, wikepidia.
import pandas as pd
# reading the data as pandasdataframe
data= pd.read_csv('training.csv')
df= pd.DataFrame(data)
# listing the columns
lisofcol=list(df.columns)
# removing form list of cloumns
lisofcol.remove('14')
#  method for pre-processing data
#  to find the sume of all the cound in the final column
def unique_sum(data_frame,colName):
    unique_value=data_frame[colName].unique()
    result=[]
    for val in unique_value:
        result.append((data_frame[colName]==val).sum())
    return result
count= unique_sum(df,'14')
#  grouping by the sum of that column
df=df.groupby(['14'])[lisofcol].sum()
# adding the count column in that data
df['count']= count
# saving the counttabpe file
df.to_csv("Ucountable.csv",index=False)

# df3.to_csv("eta0.01mue0.001percent0.79746.csv", index=False)