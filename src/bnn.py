import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchbnn as bnn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
import shap
from torch.autograd import Variable
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# These two functions are imported from /.src/linear.py used for the linear case
from src.linear import CV, mix_per


def saf_BNN (name, mr, k, n_epochs, nr, plim, ticks, title, unit):
    train, test= CV('./data/'+name+'.csv',k, 42)
    y_pred_all = np.zeros((k*len(test[0]), nr))
    y_std_all = np.zeros((k*len(test[0]), nr))
    mape20 = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for w in range(nr):
        # BNN model
        model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=len(mr), out_features=32),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=32, out_features=16),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=8),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=4),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=1),
                          )

        model.to(device)
        # BNN loss function 
        # Kullback–Leibler divergence
        mse_loss = nn.MSELoss()
        kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
        kl_weight = 0.01
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        
        Mape = []
        y_pred_k = []
        y_std_k = []
        for i in range (k):
            train, test= CV('./data/'+name+'.csv', k, 42)
            a = train[i]['ID']
            b = test[i]['ID']
            # Make sure that all train and test samples are unique 
            assert not set(a) & set(b), f"Vectors are not disjoint. Common values found: {set(a) & set(b)}"
            X = np.array(train[i][mr])
            X_scaler = StandardScaler()
            X = X_scaler.fit_transform(X)       
            X_test = np.array(test[i][mr])
            X_test = X_scaler.transform(X_test)
            X_train = torch.tensor(X, device=device, dtype=torch.float32)
            y = np.array(train[i][name]).reshape(-1,1)
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y)
            y_train = torch.tensor(y, device=device, dtype=torch.float32)
            X_test =  torch.tensor(X_test, device=device, dtype=torch.float32)
            y_test = np.array(test[i][name]).reshape(-1,1)
        # Learning 
            best_cost = np.inf   # init to infinity
            best_weights = None
            history = []
            for step in range(n_epochs):
                pre = model(X_train)
                mse = mse_loss(pre, y_train)
                kl = kl_loss(model)
                cost = mse + kl_weight*kl
                history.append(float(cost))
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                #print ("Epoch =", step)
                #print ("loss =", float(cost))
                if cost < best_cost:
                    best_cost = cost
                    best_weights = copy.deepcopy(model.state_dict())
            print('- MSE : %2.5f, KL : %2.5f' % (mse.item(), kl.item()))
            model.load_state_dict(best_weights)
            # Predictions 
            results = np.array([model(X_test).cpu().data.numpy() for k in range(10000)]) 
            results = results[:,:,0]    
            results = results.T
            results = y_scaler.inverse_transform(results)
            mean = np.array([results[i].mean() for i in range(len(results))])
            std = np.array([results[i].std() for i in range(len(results))])
            y_pred_k.append(mean)
            y_std_k.append(std)
            Mape.append(100*mean_absolute_percentage_error(y_test, mean))
            
        for s in range(k):
            y_pred_all[s*len(mean):(s+1)*len(mean), w] = y_pred_k[s]
            y_std_all[s*len(mean):(s+1)*len(mean), w] = y_std_k[s]
                
        mape20.append(np.array(Mape).mean())
        print (mape20[w])
        
        print("run =", str(w+1)+'/'+str(nr),'done')
           
    
    # ================== Regression Plot ==================
    p = Path('./figures/bnn/'+name)
    p.mkdir(parents=True, exist_ok=True)
    y_pred_all = y_pred_all.mean(axis=1)
    y_std_all = y_std_all.mean(axis=1)
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
    plt.xticks(ticks)
    plt.yticks(ticks)
    plt.title(title+' '+unit)
    train, test = CV('./data/'+name+'.csv', k, 42)
    for j in range (k):
        y_test = test[j][name]
        y_pred = np.array(test[j][mr])
        y_calb = y_pred_all[j*len(y_test):(j+1)*len(y_test)]
        y_std = y_std_all[j*len(y_test):(j+1)*len(y_test)]
        y_pred = np.mean(y_pred, axis=1)
        plt.scatter(y_pred,y_test,color='black')
        plt.errorbar(y_calb, y_test, yerr=y_std, fmt='o', color='tab:blue')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        if (j == (k-1)):
                plt.scatter(y_pred, y_test,color='black', label='$MAPE_{Ori}$ = '+"{:.2f}".format(Mape_avg)+'%')
                plt.errorbar(y_calb, y_test, yerr=y_std, fmt='o', color='tab:blue',
                         label='$MAPE_{Calib}$ = '+"{:.2f}".format(np.array(mape20).mean())+'%')
                plt.legend(fontsize=14)
                plt.savefig('./figures/bnn/'+name+'/'+name+'_bnn.png', dpi=500, bbox_inches='tight') 
    
    # ================== Iterations metrics ==================
    p = Path('./coeff/bnn/')
    p.mkdir(parents=True, exist_ok=True)
    mape20.append(np.array(mape20).mean())
    mape20.append(np.array(mape20).std())
    mape_df = pd.DataFrame(np.array(mape20), columns=["MAPE"])
    mape_df.to_csv('./coeff/bnn/'+name+'_mape.csv')

#======================================================================================================================
# SHAP analysis               
def bnn_shap (name, mr, k, n_epochs, nr):
    shap_all = np.zeros((len(mr), k))
    shap_All = np.zeros((len(mr), nr)) 
    for w in range(nr):
        for i in range (k):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = nn.Sequential(bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=len(mr), out_features=32),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=32, out_features=16),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=16, out_features=8),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=8, out_features=4),
                          nn.ReLU(),
                          bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=4, out_features=1),
                          )

            model.to(device)
            # BNN loss function 
            # Kullback–Leibler divergence
            mse_loss = nn.MSELoss()
            kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)
            kl_weight = 0.01
            # Optimizer
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            train, test= CV('./data/'+name+'.csv',k,42)
            X = np.array(train[i][mr])
            X_scaler = StandardScaler()
            X = X_scaler.fit_transform(X)
            X_train = torch.tensor(X, device=device, dtype=torch.float32)
            y = np.array(train[i][name]).reshape(-1,1)
            y_scaler = StandardScaler()
            y = y_scaler.fit_transform(y)
            y_train = torch.tensor(y, device=device, dtype=torch.float32)
            # Learning 
            best_cost = np.inf   # init to infinity
            best_weights = None
            history = []
            for step in range(n_epochs):
                pre = model(X_train)
                mse = mse_loss(pre, y_train)
                kl = kl_loss(model)
                cost = mse + kl_weight*kl
                history.append(float(cost))
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
            if cost < best_cost:
                best_cost = cost
                best_weights = copy.deepcopy(model.state_dict())
    
            print('- MSE : %2.2f, KL : %2.2f' % (mse.item(), kl.item()))
            model.load_state_dict(best_weights)
            # Get features
            test_features_df = test[i][mr] # pandas dataframe
            # Define function to wrap model to transform data to tensor
            f = lambda x: model.cpu()(Variable(torch.from_numpy(x))).detach().numpy()
            # Convert my pandas dataframe to numpy
            data = test_features_df.to_numpy(dtype=np.float32)
            data = X_scaler.fit_transform(data)
            # The explainer doesn't like tensors, hence the f function
            explainer = shap.KernelExplainer(f, data)
            # Get the shap values from my test data
            shap_values = explainer.shap_values(data)
            shap_v = np.absolute(shap_values[:,:,0]).mean(axis=0)
            shap_all [:,i] = shap_v
            shap_all_m = shap_all.mean(axis=1)
        shap_All[:,w] = shap_all_m
        print("run =", str(w+1)+'/'+str(nr),'done')
    
    p = Path('./coeff/bnn')
    p.mkdir(parents=True, exist_ok=True)
    shap_df = pd.DataFrame(shap_All, index=mr)
    shap_df.to_csv('./coeff/bnn/'+name+'_shap.csv')
    # Plots
    print ('Training is done')
    p = Path('./figures/bnn/'+name)
    p.mkdir(parents=True, exist_ok=True)
    shap_all_p = shap_All.mean(axis=1)
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.barh(mr, shap_all_p, color='lightcoral')
    plt.ylabel ('Property Method')
    plt.xlabel ('mean(|SHAP Values|)')
    plt.savefig('./figures/bnn/'+name+'/'+'shap_'+name+'.png', dpi=500, bbox_inches='tight')
    