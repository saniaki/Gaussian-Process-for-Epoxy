# Gaussian Process for 3501-6 Epoxy Resin

The rate degree of cure mean value and variance is predicted at test points using Gaussian process and some trianing data (from material model).


The method is presented in the following paper for non-parametric models: <br>
Uncertinty Qualification of Material Models for Process Simulation, O. Fernlund et al., (2020) SAMPE VIRTUAL SERIES https://www.nasampe.org/page/2020VirtualSeries <br> 

Three temperature cycle can be considered: 
  
1. isothermal (to be updated)
2. constant rate 
3. one hold cycle (to be updated)

Independent variabes are: 

1. degree of cure
2. temperature

Dependent variable is:
1. rate of degree of cure


Rate of degree of cure mean value predictions
<p align="center">
<img  align="center" src="https://github.com/saniaki/Gaussian-Process-for-Epoxy/blob/master/images/rod_mean.jpg" width="500"/>
  
  
  Rate of degree of cure variance predictions
<p align="center">
<img  align="center" src="https://github.com/saniaki/Gaussian-Process-for-Epoxy/blob/master/images/rod_variance.jpg" width="500"/>
