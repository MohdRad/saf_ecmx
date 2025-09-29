from src.visual import error_hist


# Density   
# MCMC 
error_hist(data_path='./data/hist/density_mcmc.csv', 
           mape = 0.57, 
           name='density', 
           title='MCMC/LR, $\\rho$',
           xlim=[0, 3.3],
           ylim=[0, 45],
           xticks=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
           yticks=[0, 10, 20, 30, 40])
# BNN
error_hist(data_path='./data/hist/density_bnn.csv', 
           mape = 0.42, 
           name='density_bnn', 
           title='BNN, $\\rho$',
           xlim=[-0.005,2.1],
           ylim=[0,53],
           xticks=[0, 0.5, 1.0, 1.5, 2.0],
           yticks=[0, 10, 20, 30, 40, 50])

# ============================================================================
# Viscosity
# MCMC
error_hist(data_path='./data/hist/viscosity_mcmc.csv', 
           mape = 9.02, 
           name='viscosity', 
           title='MCMC/LR, $\\nu$',
           xlim=[0,80],
           ylim=[0,50],
           xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80],
           yticks=[0, 10, 20, 30, 40, 50])

# BNN
error_hist(data_path='./data/hist/viscosity_bnn.csv', 
           mape = 6.79, 
           name='viscosity_bnn', 
           title='BNN, $\\nu$',
           xlim=[0, 55],
           ylim=[0, 55],
           xticks=[0, 10, 20, 30, 40, 50],
           yticks=[10, 20, 30, 40, 50])

# ============================================================================
# Flash Point
# MCMC 
error_hist(data_path='./data/hist/fp_mcmc.csv', 
           mape = 5.83, 
           name='flash_point', 
           title='MCMC/LR, $T_f$',
           xlim=[0,85],
           ylim=[0,50],
           xticks=[0, 10, 20, 30, 40, 50, 60, 70, 80],
           yticks=[0, 10, 20, 30, 40, 50])
# BNN
error_hist(data_path='./data/hist/fp_bnn.csv', 
           mape = 5.51, 
           name='flash_point_bnn', 
           title='BNN, $T_f$',
           xlim=[-0.2,55],
           ylim=[0,45],
           xticks=[0, 10, 20, 30, 40, 50],
           yticks=[10, 20, 30, 40])