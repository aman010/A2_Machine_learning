from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns
import numpy as np 
import mlflow 


class LinearRegression(object):
    
    #in this class, we add cross validation as well for some spicy code....
    kfold = KFold(n_splits=3)
            
    def __init__(self, _names,eta,reg ,regularization=False, xaviar = False, lr=0.001 ,method='batch', num_epochs=50, batch_size=50, cv=kfold):
        self.lr         = lr
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.method     = method
        self.cv         = cv
        self.regularization = regularization
        self._names = _names
        self.xaviar = xaviar
        self.eta = eta
        if regularization == True:
            self.reg_ = reg_
        #should be outside to get momentun effect
        if regularization == False:
            self.alpha = np.zeros(self._names.shape[0])

    def mse(self, ytrue, ypred):
        #return mean_squared_error(ytrue, ypred)
        #running a sto is causing single varibale to be measured
        if self.method == 'sto':
            #print(np.array(ytrue).shape)
            return ((ypred - ytrue) ** 2).sum() / np.array([ytrue]).shape[0]
        else:
            return ((ypred - ytrue) ** 2).sum() / ytrue.shape[0]
    
    def r_square(self, ytrue, ypred):
       ssr = sum((ytrue - ypred)**2)
       sst =sum((ytrue - np.mean(ytrue))**2)
       return 1-(ssr/sst)
    
    def fit(self, X_train, y_train):
            
        #create a list of kfold scores
        self.scores= {'val_score':[], 'train_score':[], 'r_sq_val':[], 'r_sq_train':[]}
        
        #reset val loss
        self.val_loss_old = np.infty
        
        #kfold.split in the sklearn.....
        #5 splits
        if self.xaviar == True:
            self.theta = self._initializer(X_train.shape[1]) 
        else:
            print(X_train.shape)
            self.theta = np.zeros(X_train.shape[1])

        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train)):
            
            X_cross_train = X_train[train_idx]
            y_cross_train = y_train[train_idx]
            X_cross_val   = X_train[val_idx]
            y_cross_val   = y_train[val_idx]
            
            
            #define X_cross_train as only a subset of the data
            #how big is this subset?  => mini-batch size ==> 50
            
            #one epoch will exhaust the WHOLE training set
            with mlflow.start_run(run_name=f"Fold-{fold}", nested=True):
                
                if self.eta:
                    params = {"method": self.method, "lr":self.lr, "eta":self.eta, "reg":type(self).__name__}
                    mlflow.log_params(params=params)
                

                elif self.xaviar:
                    params = {'method':self.method, 'lr':self.lr, "xaviar":self.xaviar, "reg":type(self).__name__}
                    mlflow.log_params(params=params)
                    
                elif self.regularization==True:
                    params = {"method": self.method, "lr": self.lr, "reg": self.reg_}

                
                else:
                    params = {"method": self.method, "lr": self.lr, "reg": type(self).__name__}
                    mlflow.log_params(params=params)
                    
                

                for epoch in range(self.num_epochs):
                
                    #with replacement or no replacement
                    #with replacement means just randomize
                    #with no replacement means 0:50, 51:100, 101:150, ......300:323
                    #shuffle your index
                   perm = np.random.permutation(X_cross_train.shape[0])
                  
                   
                   X_cross_train = X_cross_train[perm]
                   y_cross_train = y_cross_train[perm]
                   
                   if self.method == 'sto':
                       for batch_idx in range(X_cross_train.shape[0]):
                           X_method_train = X_cross_train[batch_idx].reshape(1, -1) #(11,) ==> (1, 11) ==> (m, n)
                           y_method_train = y_cross_train[batch_idx] 
                           train_loss = self._train(X_method_train, y_method_train) 
                   elif self.method == 'mini':
                        for batch_idx in range(0, X_cross_train.shape[0], self.batch_size):
                            #batch_idx = 0, 50, 100, 150
                            X_method_train = X_cross_train[batch_idx:batch_idx+self.batch_size, :]
                            y_method_train = y_cross_train[batch_idx:batch_idx+self.batch_size]
                            train_loss = self._train(X_method_train, y_method_train)
                   else:
                        X_method_train = X_cross_train
                        y_method_train = y_cross_train
                        train_loss = self._train(X_method_train, y_method_train)

                   mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)

                   yhat_val = self.predict(X_cross_val)
                   val_loss_new = self.mse(y_cross_val, yhat_val)
                    
                   yhat_train = self.predict(X_cross_train)
                   val_train = self.mse(y_cross_train ,yhat_train)
                    
                   r_sq = self.r_square(y_cross_val, yhat_val)
                   r_sq_train = self.r_square(y_cross_train, yhat_train)
                   mlflow.log_metric(key="val_loss", value=val_loss_new, step=epoch)
                   mlflow.log_metric(key="r_sq", value=r_sq, step=epoch)
                   mlflow.log_metric(key="r_sq_train", value=r_sq_train)
                   
                
                    #early stopping
                   if np.allclose(val_loss_new, self.val_loss_old):
                        break
                    
                   self.val_loss_old = val_loss_new
                
                   r_sq = self.r_square(y_cross_val, yhat_val)
                    
            self.scores['val_score'].append(val_loss_new)
            self.scores['r_sq_val'].append(r_sq)
            self.scores['r_sq_train'].append(r_sq_train)
            self.scores['train_score'].append(val_train)
                #print(f"Fold {fold}: {val_loss_new}")
        return self.scores
            
                    
    def _train(self, X, y):
        'eta:momentum'
        
        yhat = self.predict(X)
        m    = X.shape[0]        
        
        if self.regularization == True:
            grad = (1/m) * X.T @(yhat - y) + self.regularization.derivation(self.theta)
        else:
            grad = (1/m) * X.T @(yhat - y)
            
            #this has to updtaed in order to runregularisztion
        if self.eta and self.regularization==False:
            #raise exception eta should be between 0,1
            #alpga is velocity
            self.alpha = self.eta * self.alpha + self.lr * grad
            self.theta = self.theta - self.alpha
        else:
            self.theta = self.theta - self.lr * grad
        return self.mse(y, yhat)
    
    def _initializer(self ,sample_size):
        'sample_size: size of weight currebtly only 1D array'
        'init as initializer type could be Xavier or zero'
        #genrate random samples
        #xaviar initialization works for gradient vanising problem where the derviative variance
        #becomes constant, training goes for longer
        m = np.random.uniform(0,sample_size-1,size=sample_size).astype(int) 
        lower = -1/np.sqrt(sample_size)
        upper = 1/np.sqrt(sample_size)
                
        w=np.random.uniform(lower , upper, sample_size)
        w = w[m]
        return w
    
    def predict(self, X):
        return X @ self.theta  #===>(m, n) @ (n, )
    
    def _coef(self):
        return self.theta #remind that theta is (w0, w1, w2, w3, w4.....wn)
                               #w0 is the bias or the intercept
                               #w1....wn are the weights / coefficients / theta
    def _bias(self):
        return self.theta[0]
    
    def feature_importance(self, coefs, X_train):
        coefs = pd.DataFrame(
        coefs, columns=["Coefficients"], index=X_train.columns) 
        sns.axes_style('white')
        sns.set_style('white')

        y=coefs['Cofficients'].values 
        x = coefs.index.values
        
        colors = ['pink' if _y >=0 else 'red' for _y in y]
        
        ax = sns.barplot(x, y, palette=colors)

        for n, (label, _y) in enumerate(zip(x, y)):
            ax.annotate(
                s='{:.1f}'.format(abs(_y)),
                xy=(n, _y),
                ha='center',va='center',
                xytext=(0,10),
                textcoords='offset points',
                color=colors,
                weight='bold')

        ax.annotate(
            s=label,
            xy=(n, 0),
            ha='center',va='center',
            xytext=(0,10),
            textcoords='offset points',)  
        # axes formatting
        ax.set_yticks([])
        ax.set_xticks([])
        sns.despine(ax=ax, bottom=True, left=True)
        
        
        
    
    
class LassoPenalty:
    
    def __init__(self, l):
        self.l = l # lambda value

    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.abs(theta))
        
    def derivation(self, theta):
        return self.l * np.sign(theta)
    
class RidgePenalty:
    
    def __init__(self, l):
        self.l = l
        
    def __call__(self, theta): #__call__ allows us to call class as method
        return self.l * np.sum(np.square(theta))
        
    def derivation(self, theta):
        return self.l * 2 * theta
    
class ElasticPenalty:
    
    def __init__(self, l = 0.1, l_ratio = 0.5):
        self.l = l 
        self.l_ratio = l_ratio

    def __call__(self, theta):  #__call__ allows us to call class as method
        l1_contribution = self.l_ratio * self.l * np.sum(np.abs(theta))
        l2_contribution = (1 - self.l_ratio) * self.l * 0.5 * np.sum(np.square(theta))
        return (l1_contribution + l2_contribution)

    def derivation(self, theta):
        l1_derivation = self.l * self.l_ratio * np.sign(theta)
        l2_derivation = self.l * (1 - self.l_ratio) * theta
        return (l1_derivation + l2_derivation)
    
class Lasso(LinearRegression):
    
    def __init__(self, method, lr, reg_, l, regularization=True):
        self.regularization = LassoPenalty(l)
        super().__init__(self.regularization, lr, method, reg_)
        #mlflow.log_params(params = {'lr':self.lr, 'l':l,'method':self.method, 'reg':type(self).__name__+str(l)})

class Ridge(LinearRegression):
    
    def __init__(self, method, lr,reg_, l, regularization=True, ):
        self.regularization = RidgePenalty(l)
        super().__init__(self.regularization, lr, method, reg_)
        #mlflow.log_params(params = {'lr':self.lr, 'l':l,'method':self.method, "reg":type(self).__name__+str(l)})

        
class ElasticNet(LinearRegression):
    
    def __init__(self, method, lr, l,reg_, regularization=True, l_ratio=0.5):
        self.regularization = ElasticPenalty(l, l_ratio)
        super().__init__(self.regularization, lr, method, reg_)
        #mlflow.log_params(params = {'lr':self.lr, 'l':l,'method':self.method, "reg":type(self).__name__+str(l)})


    

    