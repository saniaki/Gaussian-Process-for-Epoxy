# Gaussian-Process-for-Epoxy
Gaussian Process in Python for 3501-6 Epoxy Material Model

Same method is presented in following paer for non-parametric models is used here:
Uncertinty Qualification of Material Models for Process Simulation, Fernlund et al., 2020, Internatial n SAMPE Technical Conference

Three temperature cycle can be considered: 
  
1. isothermal (to be updated)
2. constant rate 
3. one hold cycle (to be updated)

Independent variabes are: 

1. degree of cure
2. temperature

Dependent vatiable is:
1. rate of degree of cure

A parametric prior is 

\begin{itemize}
\item $T = T(x, t)$ is the temperature (K);
\item $t$ is time (s);
\item $\rho$ is density ($Kg/m^3$);
\item $C_p$ is specific heat capacity (J/K);
\item $k$ is thermal conductivity (W/(m*K));
\item $\dot{Q} = \dot{Q}(t)$ is heat generation within composite (W);
\item $\alpha = \alpha(t) = \frac{k}{\rho\, C_p}$ is thermal diffusivity.
\end{itemize}
