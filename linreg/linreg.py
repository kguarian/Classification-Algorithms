import numpy as np
import sklearn as sk
import pandas as pd
import scipy as sp


df = pd.read_csv("USA_Housing.csv")
X = df.iloc[:,[0,1,2,4,5]].to_numpy()

y=df.iloc[:,5].to_numpy()

#A=np.zeros(5000)

for i in range(0,len(X)):
    X[i][4]=1


#print(A.shape)

#A=np.append(A,X,axis=1)


def main(A,y):

    AT = np.transpose(A)

    print(AT.shape)
    print(y.shape)

    print(y)

    AAT = AT @ A

    result = np.linalg.solve(AAT,AT @ y)
    print(result)
    return result



v = main(X,y) #change 2nd X

print("approximation: "+str(v[0:4] @ [X[0,0],X[0,1],X[0,2],X[0,3]]+v[4])+", true value: "+ str(y[0]))