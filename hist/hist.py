import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
