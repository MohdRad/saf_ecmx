import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import pytensor.tensor as pt
from pathlib import Path

# These two functions are imported from /.src/linear.py used for the linear case
from src.linear import CV, mix_per


def polyn(name, mr, k, intercept, alpha_mu, alpha_sd, beta_mu, beta_sd, ns, ncores, plim, unit):
    Mape = []
    idatas = []
    y_pred_k = []
    y_std_k = []
    for i in range (0,k):
        train, test = CV('./data/'+name+'.csv', k, 42)
        a = train[i]['ID']
        b = test[i]['ID']
        # Make sure that all train and test samples are unique 
        assert not set(a) & set(b), f"Vectors are not disjoint. Common values found: {set(a) & set(b)}"
        X = np.array(train[i][mr])
        X_scaler = StandardScaler()
        X = X_scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=mr)
        y = np.array(train[i][name]).reshape(-1,1)
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y)
        y = pd.DataFrame(y, columns=[name])
        y = y.iloc[:,0]
        with pm.Model(coords={"predictors": X.columns.values}) as basic_model:
            # Priors for unknown model parameters
            if (intercept):
                alpha = pm.Normal("alpha", mu=alpha_mu, sigma=alpha_sd)
            beta = pm.Normal("beta", mu=beta_mu, sigma=beta_sd, shape=len(mr), dims="predictors")
            beta1 = pm.Normal("beta1", mu=beta_mu, sigma=beta_sd, shape=len(mr), dims="predictors1")
            beta2 = pm.Normal("beta2", mu=beta_mu, sigma=beta_sd, shape=len(mr), dims="predictors2")
            sigma = pm.Normal("sigma", mu=0, sigma=25)
            # Likelihood (sampling distribution) of observations
            if (intercept):
                y_obs = pm.Normal("Y_obs", mu=alpha + pt.dot(X.values, beta) + pt.dot(X.values**2, beta1) + pt.dot(X.values**3, beta2), sigma=sigma, observed=y.values)
            else:
                y_obs = pm.Normal("Y_obs", mu= pt.dot(X.values, beta) + pt.dot(X.values**2, beta1)+ pt.dot(X.values**3, beta2), sigma=sigma, observed=y.values)
            
            idata = pm.sample(ns, tune=ns+3000, chains=ncores, cores=ncores, target_accept=0.99)
        idatas.append(idata)
        X_test = test[i][mr]
        X_test = X_scaler.transform(X_test)
        y_test = np.array(test[i][name]).reshape(-1,1)
        if (intercept):
            summary = az.summary(idata, var_names=['alpha', 'beta', 'beta1', 'beta2'])
            y_pred = summary['mean'][0]
            beta = summary['mean'][1:len(mr)+1].values
            beta1 = summary['mean'][len(mr)+1:2*len(mr)+1].values
            beta2 = summary['mean'][2*len(mr)+1:].values
        else:
            summary = az.summary(idata, var_names=['beta', 'beta1', 'beta2'])
            y_pred = 0
            beta = summary['mean'][0:len(mr)].values
            beta1 = summary['mean'][len(mr):2*len(mr)].values
            beta2 = summary['mean'][2*len(mr):].values
        
        p = Path('./coeff/poly')
        p.mkdir(parents=True, exist_ok=True)    
        summary.to_csv('./coeff/poly/'+name+'_fold_'+str(i)+'.csv')
        j = 0
        while (j<len(mr)):
            y_pred +=  beta[j]*X_test[:,j]
            y_pred += beta1[j]*X_test[:,j]**2
            y_pred += beta2[j]*X_test[:,j]**3
            j=j+1
        y_scaler = StandardScaler()
        y_scaler.fit(y_test.reshape(-1,1))
        y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1))
        print ('MAPE '+'('+str(i)+'),', 100*mean_absolute_percentage_error(y_test, y_pred))
        Mape.append(100*mean_absolute_percentage_error(y_test, y_pred))
        mean = summary['mean'].values
        std = summary['sd'].values
        beta = []
        if (intercept):
            alpha = np.random.normal(mean[0], std[0], ns)
            y_dist = alpha
            for s in range(1,len(mr)+1):
                beta.append(np.random.normal(mean[s], std[s], ns))
        else:
            y_dist = 0
            for s in range(0,len(mr)):
                beta.append(np.random.normal(mean[s], std[s], ns))
        y = []
        for j in range (len(X_test)):
            for s in range(len(mr)):
                y_dist = y_dist + beta[s]*X_test[j,s]
            #y.append(y_dist)
            y.append(y_scaler.inverse_transform(y_dist.reshape(-1,1)))
            if (intercept):
                y_dist = alpha
            else:
                y_dist = 0
        y_std = []
        for s in range(len(y)):
            y_std.append(y[s].std())

        y_pred_k.append(y_pred)
        y_std_k.append(y_std)
        
    print(np.array(Mape).mean())
    
    # ================== Regression Plot ==================
    p = Path('./figures/poly/'+name)
    p.mkdir(parents=True, exist_ok=True)
    Mape_avg = mix_per('./data/'+name+'.csv', name, mr, k)
    data = pd.read_csv('./data/'+name+'.csv')
    y_true = np.array(data[name])
    y_true_mean = np.mean(y_true)
    maxx = int(np.max(y_true))
    y_true = np.insert(y_true,0,0)
    y_true = np.insert(y_true,-1,maxx+10000)
    plt.figure()
    plt.plot(y_true, y_true, color='r')
    plt.plot([y_true_mean, y_true_mean], [-10000,10000], color='r', linestyle='dashed')
    plt.xlim(plim)
    plt.ylim(plim)
    plt.title(name+' '+unit)
    train, test = CV('./data/'+name+'.csv', k, 42)
    for j in range (k):
        y_test = test[j][name]
        y_pred = np.array(test[j][mr])
        y_calb = y_pred_k[j].flatten()
        y_std = y_std_k[j]
        y_pred = np.mean(y_pred, axis=1)
        plt.scatter(y_pred,y_test,color='black')
        plt.errorbar(y_calb, y_test, yerr=y_std, fmt='o', color='tab:blue')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if (j == (k-1)):
            plt.scatter(y_pred, y_test,color='black', label='Original'+', MAPE = '+"{:.3f}".format(Mape_avg)+'%')
            plt.errorbar(y_calb, y_test, yerr=y_std, fmt='o', color='tab:blue', 
                     label='Calibrated'+', MAPE = '+"{:.3f}".format(np.array(Mape).mean())+'%')
            plt.legend()
            if (intercept):
                plt.savefig('./figures/poly/'+name+'/'+name+'_poly'+'_calib_inter.png', dpi=500, bbox_inches='tight')
            else:
                plt.savefig('./figures/poly/'+name+'/'+name+'_poly'+'_calib.png', dpi=500, bbox_inches='tight')

        