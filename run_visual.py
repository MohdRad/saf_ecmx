from src.visual import mixing_rules_plot




mr=['BIOIDEAL', 
    'BWRS', 
    'MXBONNEL', 
    'HYSPR', 
    'BWR-LS', 
    'NRTL']

mixing_rules_plot('./data/density.csv', 'density', mr, [0, 1, 2, 3] ,'Mass Density ($\\rho$)')

mr = ['BIOIDEAL', 
      'BWR-LS', 
      'MXBONNEL', 
      'HYSPR', 
      'SR-POLAR', 
      'PR-BM'] 

mixing_rules_plot('./data/viscosity.csv', 'viscosity', mr, [0,5,10,15,20,25],'Kinematic Viscosity ($\\nu$)')


mr = ['API', 
      'PM', 
      'R1', 
      'R4', 
      'TAG']

mixing_rules_plot('./data/flash_point.csv', 'flash_point', mr, [0,5,10,15], 'Flash Point ($T_f$)')
