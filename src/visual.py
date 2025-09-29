import pandas as pd 
import numpy as np 
from sklearn.metrics import  mean_absolute_percentage_error, mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path

def mixing_rules_plot (path, prop, mr, xtics, title):
    den = pd.read_csv(path)
    exp = den[prop]
    Mape = []
    for i in range (len(mr)):
        Mape.append(100*mean_absolute_percentage_error(exp, den[mr[i]]))
    df_mape = pd.DataFrame(Mape, index=mr, columns=['Error'])
    df_mape = df_mape.sort_values(by='Error', ascending=False)
    p = Path('figures/mixing_rules')
    p.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.barh(df_mape.index, width=df_mape['Error'])
    plt.xlabel('% MAPE', fontsize=18)
    plt.ylabel('Property Method', fontsize=18)
    plt.xticks(xtics, fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=18)
    plt.savefig('./figures/mixing_rules/MAPE_'+prop+'.png', dpi=500, bbox_inches='tight')
    
def error_hist (data_path, mape, name, title, xlim, ylim, xticks, yticks):
    global data
    data = pd.read_csv(data_path)
    error = data['Error']
    p = Path('./figures/histogram')
    p.mkdir(parents=True, exist_ok=True)
    global hist, bin_edges
    hist, bin_edges = np.histogram(error, bins='auto')
    plt.rcParams.update({'font.size': 18})
    plt.figure()
    plt.hist(error, bins=len(hist), color = 'tab:red', edgecolor='black')
    plt.ylabel('Frequency')
    plt.xlabel('Relative Percent Error')
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(xticks)
    plt.yticks(yticks)
    plt.title (title)
    plt.axvline(mape, linestyle = 'dashed', color='black')
    plt.savefig('./figures/histogram/'+name+'_error_distribution.png', dpi=500, bbox_inches='tight')


def metrics (path, prop, method):
    df = pd.read_csv(path)
    y_pred = df['y_pred']
    y_true = df[prop]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    print (prop, method)    
    print ('MAE =', mae)
    print ('RMSE =', rmse)