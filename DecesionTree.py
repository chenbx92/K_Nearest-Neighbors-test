from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pydotplus

iris=load_iris()
DTC=tree.DecisionTreeClassifier(min_impurity_decrease=0,min_samples_leaf=2)
X=iris.data[:,[0,2]]
Y=iris.target

#training model
DTmodel=DTC.fit(X,Y)

#plot the decision boundary
xmin=X[:,0].min()-1
xmax=X[:,0].max()+1
ymin=X[:,1].min()-1
ymax=X[:,1].max()+1
x,y=np.meshgrid(np.arange(xmin,xmax,0.05),np.arange(ymin,ymax,0.05))
z=DTC.predict(np.c_[x.ravel(),y.ravel()])
z=z.reshape(x.shape)


#draw background
plt.figure()
cmap_background=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
plt.pcolormesh(x,y,z,cmap=cmap_background)

#draw trainingpoints
cmap_points=ListedColormap(['#FF0000','#00FF00','#0000FF'])
plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmap_points,s=10)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))
plt.show()

#test one point
print('predict one point class')
print('input characteristics:')
print('x=')
testx=input()
print('y=')
testy=input()
testz=DTC.predict([[testx,testy]])
print('result: class ')
print(testz)
print('decision path is')
print(DTC.decision_path([[testx,testy]]))
print('probability is')
print(DTC.predict_proba([[testx,testy]]))

plt.pcolormesh(x,y,z,cmap=cmap_background)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmap_points,s=10)
listcolor=['#FF0000','#00FF00','#0000FF']
plt.scatter([float(testx)],[float(testy)],c=listcolor[int(testz)],edgecolors='k',s=70)
plt.xlim((xmin,xmax))
plt.ylim((ymin,ymax))

plt.show()

with open("iris.dot",'w') as f:
    tree.export_graphviz(DTC,out_file=f)

#dot -Tpdf iris.dot -o iris.pdf