# Publication
M. I. Radaideh, M. I. Radaideh, and A. Violi, ”A Bayesian Ensemble Approach for Improved Sustainable Aviation Fuel Modeling”. Energy Conversion and Management: X. 2025, https://doi.org/10.1016/j.ecmx.2025.101287 

# Note about data used
The ASCENT data was excluded because we do not have permission to share it. Therefore, the results will not match those in the paper.  

# Installation 
The best way to run the codes is using Anaconda. Create an Anaconda environment with Python 3.11 and install the required packages using:
```bash  
conda create -n NAME python=3.11
conda activate NAME
pip install -r requirments.txt
```
Replace NAME in the first line with any name. GPU is favorable for Bayesian Neural Network (BNN) training. Check whether Nvidia-cuda was installed using: 
```bash
import torch
print(torch.cuda.is_available())
```
If this prints ```False```, you can download cuda from [Pytorch](https://pytorch.org/get-started/locally/) website.

# Bayesian Linear and Polynomial Regression
To obtain the results for **Bayesian Linear Regression** run: 
```bash
python run_lin.py
```
The coefficients for each fold are written to ```./coeff/linear/```. The figures are saved in ```./figures/linear/NAME/```. ```NAME``` is the name of the property (```density```, ```viscosity```, ```flash_point```). 

Similarly, for **polynomial regression**, run: 
```bash
python run_poly.py
```
The coefficients for each fold are written to ```./coeff/poly/```. The figures are saved in ```./figures/poly/NAME/```. ```NAME``` is the name of the property (```density```, ```viscosity```, ```flash_point```). 

# Bayesian Neural Network (BNN)
To obtain the results for **Bayesian Neural Networks** run: 
```bash
python run_bnn.py
```
The figures are saved in ```./figures/bnn/NAME/```. ```NAME``` is the name of the property (```density```, ```viscosity```, ```flash_point```). To reduce the effect of BNN stochasticity, the model was trained and tested 200 times and average of metrics and SHAP values were reported in the manuscirpt. The metrics and SHAP values for each iteration are written to ```./coeff/bnn```. The mean and standard deviation of metrics are written in the last two cells of ```./coeff/bnn/NAME_mape.csv``` files.  

### Important for BNN Results Reproducibility 
The BNN model used in ```torchbnn``` is stochastic, and different results will be obtained for each training and testing. The code runs the training and testing 200 times and calculates the metrics average to mitigate the effect of model stochasticity. Accordingly, it may not be possible to reproduce the exact numbers that appear in the manuscript, but the results will be close. 

# Other Figures
To produce Figure 2, run the following:
```bash
python run_visual.py
```
To Produce the histograms in Figures 4, 8, and 12, run the following: 
```bash
python run_hist.py
```
