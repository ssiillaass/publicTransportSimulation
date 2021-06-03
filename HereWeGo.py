from itertools import cycle
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
from scipy.stats import beta 


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

    # #Erstellt die Ankunftszeiten der Passagiere ----> Gleichverteilt
    # def initPassArrival(self):
    #     passengerArrival = []
    #     for cycle in range(self.numBusses):
    #         #Erstellung der Verteilungen innerhalb eines Headways
    #         #numPassengerPerHeadway      = 500 #TODO: Stoßzeiten und Anteile der Verteilungen
    #         passengerArrivalTimes       = rnd.rand(numPassengerPerHeadway)*self.hWay
    #         passengerArrivalTimes       = np.sort(passengerArrivalTimes)    
    #         passengerArrival[len(passengerArrival):] = (cycle*self.hWay)+passengerArrivalTimes
    #     if False: #change to False if nbot needed
    #         sns.color_palette("pastel")
    #         sns.lineplot(data=passengerArrival,label='passenger arrival',alpha=1)
    #         plt.show()
    #     return passengerArrival

    #Erstellt die Ankunftszeiten der Passagiere ----> JohnsonSB  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.johnsonsb.html
    def initPassArrival(self):
        passengerArrival = []
        for cycle in range(self.numBusses):
            #Erstellung der Verteilungen innerhalb eines Headways
            #numPassengerPerHeadway      = 500 #TODO: Stoßzeiten und Anteile der Verteilungen

            passengerArrivalTimes       = johnsonsb.rvs(a, b, scale = headway, size = numPassengerPerHeadway)
            #passengerArrivalTimes       = rnd.rand(numPassengerPerHeadway)*self.hWay

            passengerArrivalTimes       = np.sort(passengerArrivalTimes)    
            passengerArrival[len(passengerArrival):] = (cycle*self.hWay)+passengerArrivalTimes
        if False: #change to False if nbot needed
            sns.color_palette("pastel")
            sns.lineplot(data=passengerArrival,label='passenger arrival',alpha=1)
            plt.show()
        return passengerArrival

    # #Erstellt die Ankunftszeiten der Passagiere ----> Beta https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html
    # def initPassArrival(self):
    #     passengerArrival = []
    #     for cycle in range(self.numBusses):
    #         #Erstellung der Verteilungen innerhalb eines Headways
    #         #numPassengerPerHeadway      = 500 #TODO: Stoßzeiten und Anteile der Verteilungen

    #         passengerArrivalTimes       = beta.rvs(a, b, scale = headway, size = numPassengerPerHeadway)
    #         #passengerArrivalTimes       = rnd.rand(numPassengerPerHeadway)*self.hWay

    #         passengerArrivalTimes       = np.sort(passengerArrivalTimes)    
    #         passengerArrival[len(passengerArrival):] = (cycle*self.hWay)+passengerArrivalTimes
    #     if False: #change to False if nbot needed
    #         sns.color_palette("pastel")
    #         sns.lineplot(data=passengerArrival,label='passenger arrival',alpha=1)
    #         plt.show()
    #     return passengerArrival

    #Erstellt die Ankunftszeiten der Öffis
    def initVehicleArrival(self):
        busArrivals     = np.zeros(self.runtime)
        busArrivals     = np.linspace(self.hWay,self.runtime,self.numBusses, dtype='int') 
        #busArrivals[busArrivalIdx] = 1
       # print(busArrivals)
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
    #print('passenger %d arrived at %s' % (passNo, env.now)) 
    nextVeh = station.getNextVeh(env.now)

    WaitingTime = station.waitingTime.append(nextVeh - env.now)
    #print(mean(station.waitingTime))
    #MeanWaitingTime = np.mean(WaitingTime)          #Geht nicht weil welche mit dem Bus ankommen? (NonType)
    return

# def johnsonSciPy():   Hier wird JohnsonSb geplottet      #https://www.geeksforgeeks.org/python-johnson-sb-distribution-in-statistics/     So geht das halt, aber brauchen wir ja gar nciht.
#     #Parameter definieren
#     numargs = johnsonsb.numargs 
#     #a, b = 2, 15
#     rv = johnsonsb(a, b) 
#     quantile = np.arange (0.5, headway, 1) 
#     # Random Variates 
#     RVar = johnsonsb.rvs(a, b, scale = headway, size = numPassengerPerHeadway) 
#     # PDF 
#     R = johnsonsb.pdf(a, b, quantile, loc = 0, scale = 1) 
#     # Representation of rnd variates
#     distribution = np.linspace(0, np.minimum(rv.dist.b, 3)) 
#     #print("Distribution : \n", distribution) 
#     #plot = plt.plot(distribution, rv.pdf(distribution)/60) 
#     #plt.show()


#def betaSciPy():               Hier wird BetaVerteilung geplottet
# bet = np.linspace(beta.ppf(0.01, a, b),

#                 beta.ppf(0.99, a, b), 100)

# plt.plot(x, beta.pdf(bet, a, b),

#         lw=1, alpha=1, label='beta pdf')
# plt.show()


###Variables###
a, b = 2, 3.5       #Parameter der JohnsonSB und Beta (Haben natürlich unterschiedlcihen Einfluss)
runtime         = 10        #Wie lange die Simulation läuft
headway         = 5         #Headway der nächsten Öffi
numPassengerPerHeadway = 500
env             = simpy.Environment()
station         = Station(env,headway,runtime)


### 3 Runtime Processes###
for passNo in range(len(station.passArrival)): #jeden passagier durchgehen 
    #bus = int(math.floor(passengerArrival[passenger]/headway)) #nummer des busses mit dem der Gast fahren soll
    env.process(transportation(env,passNo,station))

env.run(until=runtime)

sns.color_palette("pastel")
sns.lineplot(data=station.passArrival,label='passenger arrival',alpha=1)
sns.lineplot(data=station.waitingTime,label='waiting time',alpha=1)
#sns.lineplot(data=MeanWaitingTime,label='waiting time',alpha=1)     Siehe Line 105?
plt.xlabel('Passenger No.')
plt.ylabel('ClockTime')
plt.show()