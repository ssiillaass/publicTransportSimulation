from numpy.core.fromnumeric import mean
from numpy.core.function_base import linspace
from numpy.core.numeric import ones
import simpy
import numpy.random as rnd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from simpy.core import T
from simpy.resources.resource import Request
from numpy.core.fromnumeric import mean
from numpy.core.function_base import linspace
from numpy.core.numeric import ones
import simpy
import numpy.random as rnd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from simpy.core import T
from simpy.resources.resource import Request
from scipy.stats import johnsonsb

#### 1 Functions####
class Station(object):
    #Erstellt die Haltestelle die wir Simulieren wollen
    def __init__(self,env,hWay,runtime):
        self.runtime        = runtime 
        self.hWay           = hWay
        self.numBusses      = round(runtime/hWay)
        #self.busObj         = simpy.Resource(env, 1);
        self.busCounter     = 0
        self.passArrival    = self.initPassArrival()
        self.vehArrival     = self.initVehicleArrival()
        
        self.waitingTime     = []

    #Erstellt die Ankunftszeiten der Passagiere
    def initPassArrival(self):
        passengerArrival = []
        for cycle in range(self.numBusses):
            #Erstellung der Verteilungen innerhalb eines Headways
            numPassengerPerHeadway      = 500 #TODO: Stoßzeiten und Anteile der Verteilungen
            passengerArrivalTimes       = rnd.rand(numPassengerPerHeadway)*self.hWay
            passengerArrivalTimes       = np.sort(passengerArrivalTimes)    
            passengerArrival[len(passengerArrival):] = (cycle*self.hWay)+passengerArrivalTimes
        if False: #change to False if nbot needed
            sns.color_palette("pastel")
            sns.lineplot(data=passengerArrival,label='passenger arrival',alpha=1)
            plt.show()
        return passengerArrival

    #Erstellt die Ankunftszeiten der Öffis
    def initVehicleArrival(self):
        busArrivals     = np.zeros(self.runtime)
        busArrivals     = np.linspace(self.hWay,self.runtime,self.numBusses, dtype='int') 
        #busArrivals[busArrivalIdx] = 1
        print(busArrivals)
        return busArrivals

    #Errechne den Zeitpunkt des nächsten Busses
    def getNextVeh(self,time):
        if time >= self.vehArrival[self.busCounter]:
            self.busCounter = self.busCounter+1
        nxtVeh = self.vehArrival[self.busCounter]
        return nxtVeh

def transportation(env, passNo, station):
    arrTime = station.passArrival[passNo]
    yield env.timeout(arrTime)
    print('passenger %d arrived at %s' % (passNo, env.now)) 
    nextVeh = station.getNextVeh(env.now)

    station.waitingTime.append(nextVeh - env.now)
    print(mean(station.waitingTime))
    return

def johnsoncombined(headway,samples):
    #FROM Passenger arrival rates at public transport stations Author(s):Lüthi, Marco; Weidmann, Ulrich; Nash, Andrew
    c_sd = 0.15
    c_si = 1-c_sd 
    t_hw = headway
    alpha_1 = -1.2
    alpha_2 = 1
    beta    = 0.8000000001
    yPDF_vec = np.zeros(samples)
    yCDF_vec = np.zeros(samples)
    #x_vec = linspace(0,t_hw-(t_hw/samples),samples)
    x_vec = np.arange(0,t_hw+t_hw/samples,t_hw/samples)
    x_vec = np.linspace(t_hw/samples,t_hw,samples)
    i = 0
    for x in x_vec:
        if 0<=x<beta:
            a = (c_sd/t_hw)
            b = (c_si*alpha_2*t_hw)/((x+t_hw-beta)*(beta-x)*math.sqrt(2*math.pi))
            c = math.exp(-0.5*pow((alpha_1+alpha_2*math.log((x+t_hw-beta)/(beta-x))),2))
            y = a+b*c
        elif beta<x<=t_hw:
            a = (c_sd/t_hw)
            b = (c_si*alpha_2*t_hw)/((x-beta)*(t_hw+beta-x)*math.sqrt(2*math.pi))
            c = math.exp(-0.5*pow((alpha_1+alpha_2*math.log((x-beta)/(t_hw+beta-x))),2))
            y = a+b*c
        else: 
            y = 0
        yPDF_vec[i] = y
        i = i+1

    #Calc CDF
    dx = x_vec[1]-x_vec[0]
    yCDF_vec = np.cumsum(yPDF_vec*dx)

    sns.lineplot(x=x_vec,y=yPDF_vec,label='Combined Uniform and Johnson Probability Density',alpha=1)
    sns.lineplot(x=x_vec,y=yCDF_vec,label='Combined Uniform and Johnson Cumulative Density',alpha=1)
    plt.show()

    return yPDF_vec

def johnsonSciPy():
    #Parameter definieren
    numargs = johnsonsb.numargs 
    a, b = 4.32, 3.18
    rv = johnsonsb(a, b) 

    print ("RV : \n", rv) 

    quantile = np.arange (0.01, 1, 0.1) 

    # Random Variates 
    R = johnsonsb.rvs(a, b, scale = 2, size = 10) 
    print ("Random Variates : \n", R)

    # PDF 
    R = johnsonsb.pdf(a, b, quantile, loc = 0, scale = 1) 
    print ("\nProbability Distribution : \n", R)

    # Representation of rnd variates
    distribution = np.linspace(0, np.minimum(rv.dist.b, 3)) 
    print("Distribution : \n", distribution) 

    plot = plt.plot(distribution, rv.pdf(distribution)) 
    plt.show()

###Variables###
runtime         = 120
headway         = 10

env             = simpy.Environment()
station         = Station(env,headway,runtime)

johnson = johnsoncombined(station.hWay,1000)

johnsonSciPy()

### 3 Runtime Processes###
for passNo in range(len(station.passArrival)): #jeden passagier durchgehen 
    #bus = int(math.floor(passengerArrival[passenger]/headway)) #nummer des busses mit dem der Gast fahren soll
    env.process(transportation(env,passNo,station))

env.run(until=runtime)

sns.color_palette("pastel")
sns.lineplot(data=station.passArrival,label='passenger arrival',alpha=1)
sns.lineplot(data=station.waitingTime,label='waiting time',alpha=1)
plt.show()




