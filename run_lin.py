from src.linear import saf_pymc
# Intercept =================================================
# Density 

saf_pymc (k=3,                                                               # Number of folds for k-fold cross validation
          name='density',                                                    # Name of the property, choices are: 'density', 'viscosity', and 'flash_point' 
          mr=['BIOIDEAL', 'BWRS', 'MXBONNEL', 'HYSPR', 'BWR-LS', 'NRTL'],    # Mixing rules as a list
          intercept=True,                                                    # Include intercept term (alpha) 
          alpha_mu=0,                                                        # Prior distribution mean of intercept term (alpha) 
          alpha_sd=10,                                                       # Prior distribution standard deviation of intercept term (alpha)
          beta_mu=0,                                                         # Prior distribution mean of linear regression coeffficients or weights (beta)
          beta_sd=0.1,                                                       # Prior distribution standard deviation of linear regression coeffficients or weights (beta)
          ns=10000,                                                          # Number of MCMC samples per chain/core 
          ncores=20,                                                         # Number of MCMC chains==number of CPU cores 
          plim = [740,840],                                                  # x- and y-axis limits for the regression plot 
          ticks= [740, 760, 780, 800, 820, 840],                             # x- and y-axis ticks for the regression plot
          title = "MCMC/LR, $\\rho$",                                        # plot title for the regression plot  
          unit = '[kg/m$^3$]')                                               # fuel property unit  
 
   
# Viscosity
saf_pymc (k=3, 
          name='viscosity',
          mr=['BIOIDEAL', 'BWR-LS', 'MXBONNEL', 'HYSPR', 'SR-POLAR', 'PR-BM'], 
          intercept=True, 
          alpha_mu=0, 
          alpha_sd=10, 
          beta_mu=0, 
          beta_sd=0.1, 
          ns=10000, 
          ncores=20,
          plim =[0.5,7.5],
          ticks = [1,3,5,7],
          title = "MCMC/LR, $\\nu$",
          unit = '[mm$^2$/s]')
         


# Flash Point 
saf_pymc (k=3, 
          name='flash_point',
          mr=['API', 'PM', 'R1', 'R4', 'TAG'], 
          intercept=True, 
          alpha_mu=0, 
          alpha_sd=10, 
          beta_mu=0, 
          beta_sd=0.1, 
          ns=10000, 
          ncores=20,
          plim =[10,80],
          ticks = [10, 30, 50, 70],
          title = "MCMC/LR, $T_f$",
          unit = '[$^o$C]')





'''
# Without intercept  =================================================
# No need to run; the results are the same as the intercept cases. 
saf_pymc (k=3, 
          name='density',
          mr=['BIOIDEAL', 'BWRS', 'MXBONNEL', 'HYSPR', 'BWR-LS', 'NRTL'], 
          intercept=False, 
          alpha_mu=None, 
          alpha_sd=None, 
          beta_mu=0, 
          beta_sd=0.1, 
          ns=10000, 
          ncores=64,
          plim = [740,840],
          ticks= [740, 760, 780, 800, 820, 840],
          title = "MCMC/LR, $\\rho$",
          unit = '[kg/m$^3$]')

# Viscosity
saf_pymc (k=3, 
          name='viscosity',
          mr=['BIOIDEAL', 'BWR-LS', 'MXBONNEL', 'HYSPR', 'SR-POLAR', 'PR-BM'], 
          intercept=False, 
          alpha_mu=None, 
          alpha_sd=None, 
          beta_mu=0, 
          beta_sd=0.1, 
          ns=10000, 
          ncores=64,
          plim =[0.5,7.5],
          ticks = [1,3,5,7],
          title = "MCMC/LR, $\\nu$",
          unit = '[mm$^2$/s]')


# Flash Point 
# Without intercept 
saf_pymc (k=3, 
          name='flash_point',
          mr=['API', 'PM', 'R1', 'R4', 'TAG'], 
          intercept=False, 
          alpha_mu=None, 
          alpha_sd=None, 
          beta_mu=0, 
          beta_sd=0.1, 
          ns=10000, 
          ncores=64,
          plim =[10,80],
          ticks = [10, 30, 50, 70],
          title = "MCMC/LR, $T_f$",
          unit = '[$^o$C]')
'''


