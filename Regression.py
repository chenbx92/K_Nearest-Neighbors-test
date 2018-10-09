import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import linear_model,datasets

cmtp = plt.get_cmap('Reds')
cmbg = plt.get_cmap('Greys')
cmap_bg=ListedColormap(['#FAAAAA','#F00000'])
cmap_tp=ListedColormap(['#FFFFAA','#0000AA'])
cmap_bg4=ListedColormap(['#FFAAAA','#AAFFAA','#AAAAFF'])
cmap_tp4=ListedColormap(['#FF0000','#00FF00','#0000FF'])
boston=datasets.load_boston()
X=boston.data[:,[5,7]]
Y=boston.target
#deal with Y for logistic regression
YL=np.where(Y>22,1,0)
#deal with Y for multi-class logistic regression
YM=np.where(Y<15,0,Y)
YM=np.where((YM>=15)&(YM<25),1,YM)
YM=np.where(YM>=25,2,YM)


mode=input('linear regression input 1 or logistic regression test input 2 or multinomial logistic regression test input 3:')

if mode == '1':

    LRM=linear_model.LinearRegression()
    LRM.fit(X,Y)

    #plot the prediction area
    xmin=X[:,0].min()-1
    xmax=X[:,0].max()+1
    ymin=X[:,1].min()-1
    ymax=X[:,1].max()+1
    x,y=np.meshgrid(np.arange(xmin,xmax,0.05),np.arange(ymin,ymax,0.05))
    Z=LRM.predict(np.c_[x.ravel(),y.ravel()])
    Z=Z.reshape(x.shape)

    plt.figure()
    #draw test background
    plt.pcolormesh(x,y,Z,cmap=cmbg)
    #draw training points
    plt.scatter(X[:,0],X[:,1],c=Y,cmap=cmtp,s=10)
    plt.xlabel('roomNo')
    plt.ylabel('distancetocenter')
    plt.title('coefficient of determination: %.2f' % LRM.score(X,Y))
    plt.colorbar()

    plt.show()

elif mode == '2': #actually kind of classification
    LORM=linear_model.LogisticRegression()
    LORM.fit(X,YL)

    # plot the prediction area
    xmin = X[:, 0].min() - 1
    xmax = X[:, 0].max() + 1
    ymin = X[:, 1].min() - 1
    ymax = X[:, 1].max() + 1
    x, y = np.meshgrid(np.arange(xmin, xmax, 0.05), np.arange(ymin, ymax, 0.05))
    Z = LORM.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)

    plt.figure()
    # draw test background
    plt.pcolormesh(x, y, Z, cmap=cmap_bg)
    plt.colorbar()
    # draw training points
    plt.scatter(X[:, 0], X[:, 1], c=YL, cmap=cmap_tp, s=10)
    plt.xlabel('roomNo')
    plt.ylabel('distancetocenter')
    plt.title('mean accuracy: %.2f' % LORM.score(X, YL))
    plt.colorbar()
    plt.show()

elif mode == '3':  # multinomial logistic regression/softmax
    MLORM=linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
    MLORM.fit(X,YM)

   # plot the prediction area
    xmin = X[:, 0].min() - 1
    xmax = X[:, 0].max() + 1
    ymin = X[:, 1].min() - 1
    ymax = X[:, 1].max() + 1
    x, y = np.meshgrid(np.arange(xmin, xmax, 0.05), np.arange(ymin, ymax, 0.05))
    Z = MLORM.predict(np.c_[x.ravel(), y.ravel()])
    Z = Z.reshape(x.shape)

    plt.figure()
    # draw test background
    plt.pcolormesh(x, y, Z, cmap=cmap_bg4)
    plt.colorbar()
    # draw training points
    plt.scatter(X[:, 0], X[:, 1], c=YM, cmap=cmap_tp4, s=10)
    plt.xlabel('roomNo')
    plt.ylabel('distancetocenter')
    plt.title('mean accuracy: %.2f' % MLORM.score(X, YM))
    plt.colorbar()
    plt.show()
else:
    print('no such mode')
