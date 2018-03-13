import os
os.chdir("../")

import pickle
import pandas as pd
import pickle
import os
import numpy as np
from sklearn.model_selection import KFold # import KFold

from onoff import onoff
from svgp import svgp

data = pickle.load(open("data/pptr.pickle","rb"))
Xraw = np.concatenate([data['Xtrain'],data['Xtest']])
Yraw = np.concatenate([data['Ytrain'],data['Ytest']])
Xraw[:,2] = Xraw[:,2]/1000

kf = KFold(n_splits=5, random_state=1234, shuffle=True)
nfold = 0

for train_index, test_index in kf.split(Xraw):
    nfold = nfold+1
    Xtrain, Xtest = Xraw[train_index], Xraw[test_index]
    Ytrain, Ytest = Yraw[train_index], Yraw[test_index]
    print(Ytrain.shape,Ytest.shape)

    dir = os.getcwd() + "/data/cv/"+str(nfold)+"/"

    if not os.path.exists(dir):
        os.makedirs(dir)

    data = {'Xtrain':Xtrain,'Ytrain':Ytrain,'Xtest':Xtest,'Ytest':Ytest}
    pickle.dump(data, open(dir+"data.pickle","wb"))
