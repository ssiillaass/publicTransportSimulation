from sympy import *

init_session()

#window opens in TERMINAL
#PLUG FOLLOWING CODE IN THIS WINDOW CALLED "(SymPyConsole)""
x = Symbol('x')
pdf = Symbol('pdf')
c_sd = Symbol('c_sd') 
#c_si = 1-c_sd 
t_hw = Symbol('t_hw')
alpha_1 = Symbol('alpha_1')
alpha_2 = Symbol('alpha_2')
beta_    = Symbol('beta_')

a = c_sd/t_hw
b_1 = ((1-c_sd)*alpha_2*t_hw)/((x+t_hw-beta_)*(beta_-x)*sqrt(2*pi))
c_1 = exp(-0.5*pow((alpha_1+alpha_2*log((x+t_hw-beta_)/(beta_-x))),2))

b_2 = ((1-c_sd)*alpha_2*t_hw)/((x-beta_)*(t_hw+beta_-x)*sqrt(2*pi))
c_2 = exp(-0.5*pow((alpha_1+alpha_2*log((x-beta_)/(t_hw+beta_-x))),2))

#Erster Teil der Verteilung sähe so aus
pdf_1 = a+b_1*c_1 #kann man sich angucken indem man einfach pdf_1 ins Fenster packt

integrate(a+b_1*c_1,x)

pdf = Piecewise( (a+b_1*c_1, (0<x) & (x<beta_)), (a+b_2*c_2, (beta_<x) & (x<t_hw)), (0, True))

cdf = integrate(pdf,x)
