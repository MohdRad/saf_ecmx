from src.bnn import saf_BNN, bnn_shap

# Density 

saf_BNN (name = 'density',                                                     # Name of property, choices are: 'density', 'viscosity', 'flash_point'
     mr= ['BIOIDEAL', 'BWRS', 'MXBONNEL', 'HYSPR', 'BWR-LS', 'NRTL'],          # Mixing rules as a list 
     k=3,                                                                      # Number of folds for k-fold cross validation  
     n_epochs = 1000,                                                          # Number of training epochs for Bayesian Neural Network (BNN)
     nr = 200,                                                                 # Number of runs to mitigate the effect of BNN stochasticity
     plim= [740,840],                                                          # x- and y-axis limits for the regression plot 
     ticks=[740,760,780,800,820,840],                                          # x- and y-axis ticks for the regression plot
     title = 'BNN, $\\rho$',                                                   # Plot title for the regression plot 
     unit='[kg/m$^3$]')                                                        # Fuel property unit



# Viscosity
saf_BNN (name = 'viscosity', 
     mr= ['BIOIDEAL', 'BWR-LS', 'MXBONNEL', 'HYSPR', 'SR-POLAR', 'PR-BM'], 
     k=3, 
     n_epochs = 1000,
     nr = 200,                                                                  # Number of runs to mitigate the effect of BNN stochasticity
     plim= [0.5,7.5],
     ticks=[1,3,5,7],
     title = 'BNN, $\\nu$',
     unit='[mm$^2$/s]')



# Flash Point 
saf_BNN (name = 'flash_point', 
     mr= ['API', 'PM', 'R1', 'R4', 'TAG'], 
     k=3, 
     n_epochs = 1000,
     nr = 200,                                                                  # Number of runs to mitigate the effect of BNN stochasticity
     plim= [10,75], 
     ticks=[10,30,50,70],
     title = 'BNN, $T_f$',
     unit='[$^o$C]')


#=========================================================================================================================================================
# SHAP analysis


bnn_shap(name='density',                                                         # Name of property, choices are: 'density', 'viscosity', 'flash_point'  
         mr=['BIOIDEAL', 'BWRS', 'MXBONNEL', 'HYSPR', 'BWR-LS', 'NRTL'],         # Mixing rules as a list
         k=3,                                                                    # Number of folds for k-fold cross validation
         n_epochs=1000,                                                          # Number of training epochs for Bayesian Neural Network (BNN)
         nr = 200)                                                               # Number of runs to mitigate the effect of BNN stochasticity



bnn_shap(name='viscosity', 
         mr=['BIOIDEAL', 'BWR-LS', 'MXBONNEL', 'HYSPR', 'SR-POLAR', 'PR-BM'],
         k=3,
         n_epochs=1000,
         nr = 200)
   
         

bnn_shap(name='flash_point', 
         mr=['API', 'PM', 'R1', 'R4', 'TAG'],
         k=3,
         n_epochs=1000,
         nr = 200)
