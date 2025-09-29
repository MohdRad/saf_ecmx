import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pytensor.tensor as pt
import seaborn as sns
from pathlib import Path

   
# Kfold cross validation 
def CV(path, k, seed):
    experimental = pd.read_csv(path)
    counter = shuffle([i for i in range(len(experimental))], random_state=seed)
    fold = int(len(experimental)/k)
    folds = []
    for i in range(k):
        folds.append(counter[i*fold:(i+1)*fold])
    train = []
    test = []
    for i in range(k):
        train.append(experimental.drop(index=folds[i]))
        test.append(experimental.loc[folds[i]])
    return train, test
    
# Average error of mixing rules
def mix_per (path, prop, mr,k):
    train, test = CV(path, k, 42)
    Mape_avg = []
    for mix in mr:
        mappe = []
        for j in range (k):
            y_test = test[j][prop]
            y_pred = test[j][mix]
            mappe.append(100*mean_absolute_percentage_error(y_test, y_pred))
        Mape_avg.append(np.mean(mappe))
    return np.mean(Mape_avg)


# Bayesian analysis by MCMC
def saf_pymc (k, name, mr, intercept, alpha_mu, alpha_sd, beta_mu, beta_sd, ns, ncores, plim, ticks, title, unit):
    Mape = []
    idatas = []
    summaries = []
    y_pred_k = []
    y_std_k = []
    for i in range (0,k):
        train, test = CV('./data/'+name+'.csv', k, 42)
        a = train[i]['ID']
        b = test[i]['ID']
        # Make sure that all train and test samples are unique 
        assert not set(a) & set(b), f"Vectors are not disjoint. Common values found: {set(a) & set(b)}"
        #X = cal_prop(mr, train[i], name)
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
            sigma = pm.Normal("sigma", mu=0, sigma=25)
            # Likelihood (sampling distribution) of observations
            if (intercept):
                y_obs = pm.Normal("Y_obs", mu=alpha + pt.dot(X.values, beta), sigma=sigma, observed=y.values)               
            else:
                Y_obs = pm.Normal("Y_obs", mu=pt.dot(X.values, beta), sigma=sigma, observed=y.values)
                
            
            idata = pm.sample(ns, tune=ns+2000, chains=ncores, cores=ncores, target_accept=0.99)
        idatas.append(idata)
        X_test = test[i][mr]
        X_test = X_scaler.transform(X_test)
        y_test = np.array(test[i][name]).reshape(-1,1)
        if (intercept):
            summary = az.summary(idata, var_names=['alpha', 'beta', 'sigma'])
            y_pred = summary['mean'][0]
            beta = summary['mean'][1:].values
        else:
            summary = az.summary(idata, var_names=['beta', 'sigma'])
            y_pred = 0
            beta = summary['mean'].values
        p = Path('./coeff/linear')
        p.mkdir(parents=True, exist_ok=True)    
        summary.to_csv('./coeff/linear/'+name+'_fold_'+str(i)+'.csv')
        summaries.append(summary)
        j = 0
        while (j<len(mr)):
            y_pred +=  beta[j]*X_test[:,j]
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
    p = Path('./figures/linear/'+name)
    p.mkdir(parents=True, exist_ok=True)
    Mape_avg = mix_per('./data/'+name+'.csv', name, mr, k)
    data = pd.read_csv('./data/'+name+'.csv')
    y_true = np.array(data[name])
    y_true_mean = np.mean(y_true)
    maxx = int(np.max(y_true))
    y_true = np.insert(y_true,0,0)
    y_true = np.insert(y_true,-1,maxx+10000)
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.plot(y_true, y_true, color='r')
    plt.plot([y_true_mean, y_true_mean], [-10000,10000], color='r', linestyle='dashed')
    plt.xlim(plim)
    plt.ylim(plim)
    plt.title(title+' '+unit)
    plt.xticks(ticks)
    plt.yticks(ticks)
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
            plt.scatter(y_pred, y_test,color='black', label='$MAPE_{Ori}$ = '+"{:.2f}".format(Mape_avg)+'%')
            plt.errorbar(y_calb, y_test, yerr=y_std, fmt='o', color='tab:blue', 
                         label='$MAPE_{Calib}$ = '+"{:.2f}".format(np.array(Mape).mean())+'%')
            plt.legend(fontsize=14)
            if (intercept):
                plt.savefig('./figures/linear/'+name+'/'+name+'_calib_inter.png', dpi=500, bbox_inches='tight')
            else:
                plt.savefig('./figures/linear/'+name+'/'+name+'_calib.png', dpi=500, bbox_inches='tight')
            
    
    # =================Posterior Distributions==================
    data = []
    for s in range(k):
        if (intercept):
            data.append(az.extract(data=idatas[s], group='posterior', var_names=['alpha','beta', 'sigma']))
        else:
            data.append(az.extract(data=idatas[s], group='posterior', var_names=['beta', 'sigma']))
          
    avg = []
    if (intercept):
        summ_alpha = np.zeros(ns*ncores)
        summ_beta = np.zeros(ns*ncores)
        summ_sigma = np.zeros(ns*ncores)
    else:
        summ = np.zeros(ns*ncores)
        summ_sigma = np.zeros(ns*ncores)

    if (intercept):
        for s in range(k):
            summ_alpha += data[s]['alpha']
            summ_sigma += data[s]['sigma']
        avg.append (summ_alpha/k)
        avg.append(summ_sigma/k)
        for j in range(len(mr)):
            for s in range(k):
                summ_beta +=data[s]['beta'][j]
            avg.append(summ_beta/k)
            summ_beta = np.zeros(ns*ncores)
    else:
        for s in range(k):
            summ_sigma += data[s]['sigma']
        avg.append(summ_sigma/k)
        for j in range(len(mr)):
            for s in range(k):
                summ +=data[s][j]
            avg.append(summ/k)
            summ = np.zeros(ns*ncores)
    plt.figure()
    if (intercept):
        sns.kdeplot(avg[0], label='$\\alpha$', legend=True)
        sns.kdeplot(avg[1], label='$\\sigma$', legend=True)
        for s in range(len(mr)):
            sns.kdeplot(avg[s+2], label='$\\beta_{'+mr[s]+'}$', legend=True)
        plt.title("Posterior Distributions")
        plt.xlabel("Parameter Value (Scaled)")
        plt.legend(fontsize=16)
        plt.savefig('./figures/linear/'+name+'/'+'posterior_inter_'+name+'.png', dpi=500, bbox_inches='tight')
        
    else:
        sns.kdeplot(avg[0], label='$\\sigma$', legend=True)
        for s in range(0,len(mr)):
            sns.kdeplot(avg[s], label='$\\beta_{'+mr[s]+'}$', legend=True)
        plt.title("Posterior Distributions")
        plt.xlabel("Parameter Value (Scaled)")
        plt.legend(fontsize=16)
        plt.savefig('./figures/linear/'+name+'/'+'posterior_'+name+'.png', dpi=500, bbox_inches='tight')
       
    
    #===============Forest Plot================
    summaries = []
    for s in range (k):
        summaries.append(az.summary(idatas[s], var_names=['alpha', 'sigma', 'beta']))
    
    mean = np.zeros(len(mr)+2)
    std =  np.zeros(len(mr)+2)
    
    for s in range (k):
        mean += summaries[s]['mean']
        std += summaries[s]['sd']
    
    mean/=k
    std/=k
    
    plt.figure()
    for s in range(len(mr)):
        plt.errorbar(mean[s+2], '$\\beta_{'+mr[s]+'}$', xerr=float(std[s+2]), fmt='o', color='tab:red')
    plt.errorbar(mean[0], '$\\alpha$', xerr=float(std[0]), fmt='o', color='tab:red')
    plt.errorbar(mean[1], '$\\sigma$', xerr=float(std[1]), fmt='o', color='tab:red')
    plt.title("Forest Plot")
    plt.xlabel("Parameter Value (Scaled)")
    #plt.ylabel('Property Method')
    if(intercept):
        plt.savefig('./figures/linear/'+name+'/'+'forest_inter_'+name+'.png', dpi=500, bbox_inches='tight')
    else:
        plt.savefig('./figures/linear/'+name+'/'+'forest_'+name+'.png', dpi=500, bbox_inches='tight')
     


