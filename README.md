# zero-inflated-gp
Implementation of Variational Zero Inflated Gaussian Process with Sparse Kernels

The main model is implemented in TensorFlow. All the supporting functions under onoffgp module have been insipred/partially taken from GPflow(https://github.com/GPflow/)

The Finnish precipitation dataset 'data/pptr.pickle' contains rainfall measurements across 105 observatories for the month of June 2018. Data has been taken from Finnish Meteorological Institute (http://en.ilmatieteenlaitos.fi/)

 Run create_cvsplits.py to gnerate cross-validation splits of the entire dataset
 
 Following models have been implemented :
 1. Zero-Inflated GP (onoffgp.py)
 2. GP regression (svgp.py)
 3. GP classification (scgp.py)
 3. Hurdle models (hurdle.py)
 4. Zero inflated model : GPC + GPR (zeroinflated.py)