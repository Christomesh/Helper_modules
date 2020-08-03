class XGBFunc():
    def __init__(self,train,label,test,model,params,iterations,early_stopping,verbose):
        self.train,self.label = train,label
        self.test,self.params = test,params
        self.iterations,self.early_stopping = iterations,early_stopping
        self.verbose,self.model = verbose,model
        
    def __call__(self,plot=True):
        return self.fit(plot)
    
    def fit(self,plot):
        val_p = np.zeros(len(self.train))
        scores_test = []
        scores_train = []
        test_p = np.zeros(len(self.test))
        kf = KFold(n_splits=5,shuffle=True,random_state=2000)
        for fold,(train_index,test_index) in enumerate(kf.split(self.train)):
            X_train,X_test = self.train.iloc[train_index],self.train.iloc[test_index]
            y_train,y_test = self.label.iloc[train_index],self.label.iloc[test_index]
            
            dtrain = self.model.DMatrix(data=X_train,label=y_train)
            dval = self.model.DMatrix(data=X_test,label=y_test)
            val_matrix = self.model.DMatrix(X_test)
            train_matrix = self.model.DMatrix(X_train)
            test_matrix = self.model.DMatrix(self.test)
            
            model = self.model.train(params=self.params,dtrain=dtrain,num_boost_round=self.iterations,early_stopping_rounds=self.early_stopping,evals=[(dtrain,'train'),(dval,'validation')],verbose_eval=self.verbose)
            scores_test.append(np.sqrt(mean_squared_error(y_test,model.predict(val_matrix))))
            scores_train.append(np.sqrt(mean_squared_error(y_train,model.predict(train_matrix))))
            val_p[test_index] += model.predict(val_matrix)
            test_p += model.predict(test_matrix)
        mean_train_scores = np.mean(scores_train)
        mean_val_scores = np.mean(scores_test)
        
        print(f"Training score: {mean_train_scores} ")
        print(f"Validation score: {mean_val_scores} ")
        
        if plot: self.plot_feat_imp(model)
            
        return val_p,test_p,model
    
    def plot_feat_imp(self, model):
        feat_imp = pd.Series(model.get_fscore()).sort_values(ascending=False)
        plt.figure(figsize=(12,8))
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')


class LGBFunc():
    def __init__(self,train,label,test,model,params,iterations,early_stopping,verbose):
        self.train,self.label = train,label
        self.test,self.params = test,params
        self.iterations,self.early_stopping = iterations,early_stopping
        self.verbose,self.model = verbose,model
        
    def __call__(self,plot=True):
        return self.fit(plot)
    
    def fit(self,plot):
        val_p = np.zeros(len(self.train))
        scores_test = []
        scores_train = []
        test_p = np.zeros(len(self.test))
        kf = KFold(n_splits=5,shuffle=True,random_state=2000)
        for fold,(train_index,test_index) in enumerate(kf.split(self.train)):
            X_train,X_test = self.train.iloc[train_index],self.train.iloc[test_index]
            y_train,y_test = self.label.iloc[train_index],self.label.iloc[test_index]
            
            dtrain = self.model.Dataset(data=X_train,label=y_train)
            dval = self.model.Dataset(data=X_test,label=y_test)
            
            model = self.model.train(params=self.params,train_set=dtrain,num_boost_round=self.iterations,early_stopping_rounds=self.early_stopping,valid_sets=[(dtrain),(dval)],verbose_eval=self.verbose)
            scores_test.append(np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
            scores_train.append(np.sqrt(mean_squared_error(y_train,model.predict(X_train))))
            val_p[test_index] += model.predict(X_test,num_iteration=model.best_iteration)
            test_p += model.predict(test,num_iteration=model.best_iteration)
        mean_train_scores = np.mean(scores_train)
        mean_val_scores = np.mean(scores_test)
        
        print(f"Training score: {mean_train_scores} ")
        print(f"Validation score: {mean_val_scores} ")
        
        if plot: self.plot_feat_imp(model)
            
        return val_p,test_p,model
    
    def plot_feat_imp(self, model):
        
        fi = pd.Series(index=self.train.columns, data= model.feature_importance())
        _ = plt.figure(figsize=(10, 50))
        _ = fi.sort_values().plot(kind='barh')

        
class CATFunc():
    def __init__(self,train,label,test,model,params,iterations,early_stopping,verbose):
        self.train,self.label = train,label
        self.test,self.params = test,params
        self.iterations,self.early_stopping = iterations,early_stopping
        self.verbose,self.model = verbose,model
        
    def __call__(self,plot=True):
        return self.fit(plot)
    
    def fit(self,plot):
        val_p = np.zeros(len(self.train))
        scores_test = []
        scores_train = []
        test_p = np.zeros(len(self.test))
        kf = KFold(n_splits=5,shuffle=True,random_state=2000)
        for fold,(train_index,test_index) in enumerate(kf.split(self.train)):
            X_train,X_test = self.train.iloc[train_index],self.train.iloc[test_index]
            y_train,y_test = self.label.iloc[train_index],self.label.iloc[test_index]
            
            dtrain = self.model.Pool(data=X_train,label=y_train)
            dval = self.model.Pool(data=X_test,label=y_test)
            
            model = self.model.train(params=self.params,dtrain=dtrain,num_boost_round=self.iterations,
                    early_stopping_rounds=self.early_stopping,eval_set=[(dtrain),(dval)],verbose_eval=self.verbose)
            
            scores_test.append(np.sqrt(mean_squared_error(y_test,model.predict(X_test))))
            scores_train.append(np.sqrt(mean_squared_error(y_train,model.predict(X_train))))
            val_p[test_index] += model.predict(X_test)
            test_p += model.predict(test)
        mean_train_scores = np.mean(scores_train)
        mean_val_scores = np.mean(scores_test)
        
        print(f"Training score: {mean_train_scores} ")
        print(f"Validation score: {mean_val_scores} ")
        
        if plot: self.plot_feat_imp(model)
            
        return val_p,test_p,model
    
    def plot_feat_imp(self, model):
        
        fi = pd.Series(index=self.train.columns, data= model.feature_importances_)
        _ = plt.figure(figsize=(10, 50))
        _ = fi.sort_values().plot(kind='barh')