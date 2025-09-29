from src.poly import polyn

# Polynomial

polyn (name = 'density',                                                 # Name of the property, choices are: 'density', 'viscosity', and 'flash_point'                                        
       mr = ['BIOIDEAL', 'BWRS', 'MXBONNEL', 'HYSPR', 'BWR-LS', 'NRTL'], # Mixing rules as a list
       k = 3,                                                            # Number of folds for k-fold cross validation
       intercept=True,                                                   # Include intercept term (alpha)  
       alpha_mu=0,                                                       # Prior distribution mean of intercept term (alpha) 
       alpha_sd=10,                                                      # Prior distribution standard deviation of intercept term (alpha)
       beta_mu=0,                                                        # Prior distribution mean of linear regression coefficients or weights (beta)
       beta_sd=0.1,                                                      # Prior distribution standard deviation of linear regression coefficients or weights (beta)
       ns=10000,                                                         # Number of MCMC samples per chain/core 
       ncores=20,                                                        # Number of MCMC chains==number of CPU cores 
       plim = [740,840],                                                 # x- and y-axis limits for the regression plot  
       unit = '[kg/m$^3$]')                                              # fuel property unit 



polyn (name = 'viscosity',
       mr = ['BIOIDEAL', 'BWR-LS', 'MXBONNEL', 'HYSPR', 'SR-POLAR', 'PR-BM'],
       k = 3,
       intercept=True,
       alpha_mu=0, 
       alpha_sd=10,
       beta_mu=0,
       beta_sd=0.1,
       ns=10000, 
       ncores=20,
       plim =[0.5,7.5],
       unit = '[mm$^2$/s]')



polyn (name = 'flash_point',
       mr = ['API', 'PM', 'R1', 'R4', 'TAG'],
       k = 3,
       intercept=True,
       alpha_mu=0, 
       alpha_sd=10,
       beta_mu=0,
       beta_sd=0.1,
       ns=10000, 
       ncores=20,
       plim =[10,80],
       unit = '[$^o$C]')
    