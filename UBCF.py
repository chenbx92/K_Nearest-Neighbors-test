import pandas as pd
import math

traindf=pd.DataFrame({'mova':[1,2,3,4,2],'movb':[2,4,None,None,4],'movc':[3,1,None,None,2],'movd':[2,3,4,2,3],'move':[1,2,4,5,1]})
traindf.rename(index={0:'user1',1:'user2',2:'user3',3:'user4',4:'user5'},inplace=True)
userdf=['user3','user4']
corr=traindf.T.corr()
print('pearson corr \n',corr)
rats=traindf.copy()

for userid in userdf:
    dfnull=traindf.loc[userid][traindf.loc[userid].isnull()]
    userv=traindf.loc[userid].mean()
    for i in range(len(dfnull)):
        nft=traindf[dfnull.index[i]].notnull()
        nlist=traindf[dfnull.index[i]][nft]
        nratsum=0
        corsum=0
        if(nlist.size!=0):
            nv=traindf.loc[nlist.index,:].T.mean()
            for index in nlist.index:
                ncor=corr.loc[userid,index]
                nratsum+=ncor*(traindf[dfnull.index[i]][index]-nv[index])
                corsum+=abs(ncor)
            if(corsum!=0):
                rats.at[userid,dfnull.index[i]]=userv+nratsum/corsum
            else:
                rats.at[userid,dfnull.index[i]]=userv
        else:
            rats.at[userid,dfnull.index[i]]=None

    adlist=rats.loc[userid][traindf.loc[userid].isnull()]
    aditem=adlist.idxmax()
    print('advise '+userid+' with '+aditem)
