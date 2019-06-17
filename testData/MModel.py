#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn import metrics
from sklearn import model_selection

%matplotlib notebook

# In[2]:


def getmse(field,reg,thedf,tosel):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(thedf[tosel],thedf[field], test_size=0.3)
    reg.fit(X_train,y_train)
    result=reg.predict(X_test)
    return (metrics.mean_squared_error(y_test,result), metrics.r2_score(y_test,result))

# In[3]:


df = pandas.read_csv("mmodelfile.csv")

# In[4]:


df.corr()[[x for x in df.columns if x[-1]=="C"]]

# In[5]:


maxes={}
for col in df.columns:
    if col[-1] == "C":
        continue
    if col[:-1] not in maxes:
        maxes[col[:-1]]=df.max()[col]
    if df.max()[col] > maxes[col[:-1]]:
        maxes[col[:-1]]=df.max()[col]

# In[6]:


maxes

# In[7]:


ndf=pandas.DataFrame()
for col in df.columns:
    ndf[col]=df[col]/maxes[col[:-1]]

# In[8]:


sel=[]
topred=[]
for col in ndf.columns:
    if "TIME" in col:
        continue
    if col[-1] == 'C':
        topred.append(col) 
    else:
        sel.append(col)

regs = {"SVR":svm.SVR(gamma='auto'),"MLP":MLPRegressor()}
resultsfullscale={}
for feature in topred:
    for name,_class in regs.items():
        if feature not in resultsfullscale:
            resultsfullscale[feature]={}
        resultsfullscale[feature][name]=getmse(feature,_class,ndf,sel)

resfullscaledf=pandas.DataFrame.from_dict(resultsfullscale)
resfullscaledf

# In[9]:


max_corr=ndf.corr().loc[[x for x in df.columns if x[-1]!="C"],[x for x in df.columns if x[-1]=="C"]].abs().idxmax()
factors={k:[v] for k,v in max_corr.iteritems()}

# In[10]:


def avgmser2(factorlist):
    mselist=[]
    r2list=[]
    for i in range(10):
        onemse,oner2=getmse(target,MLPRegressor(),ndf,factorlist)
        mselist.append(onemse)
        r2list.append(oner2)
    r2=sum(r2list)/len(r2list)
    mse=sum(mselist)/len(mselist)
    return (mse,r2)
        

possibles=[x for x in df.columns if x[-1]!="C" and "TIME" not in x]
print(possibles)
for target,current in factors.items():
    maxfound=False
    r2=0
    while (not maxfound) and r2 < 0.95:
        mse,r2=avgmser2(current)
        print(target,current,r2)
        bestr2=r2+0.001
        bestmse=mse
        bestnew="NONE"
        for addition in possibles:
            if addition in current:
                continue
            tempcurrent=list(current)
            tempcurrent.append(addition)
            tempmse,tempr2=avgmser2(tempcurrent)
            if tempr2 > bestr2:
                bestr2=tempr2
                bestnew=addition
                bestmse=tempmse
        if bestnew=="NONE":
            maxfound=True
        else:
            current.append(bestnew)
            r2=bestr2
            mse=bestmse
    print("target=",target,"factors=",current,"r2=",r2,"mse=",mse)
    factors[target]=current

print(factors)

# In[ ]:



