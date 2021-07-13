from enum import Enum
from os import error
import numpy as np
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import simpy
import math
from scipy.stats import beta

np.random.seed(10)

class Distribution(Enum):
    normal      = 1
    uniform     = 2
    beta        = 3
    johnson_sb  = 4
    none        = 5

class Station(object):
    # Erstellt die Haltestelle die wir Simulieren wollen
    def __init__(self, env, hWay, runtime):
        self.runtime = runtime
        self.hWay = hWay
        self.numBusses = round(runtime/hWay)
        self.busCounter = 0
        self.unpunctuality = []
        self.passArrival = self.initPassArrival()
        self.vehArrival = self.initVehicleArrival()
        self.waitingTime = []
        self.meanWaiting = []

    # Erstellt die Ankunftszeiten der Passagiere ----> Beta https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.beta.html  
    def initPassArrival(self):
        passengerArrival = []
        for cycle in range(self.numBusses):
            # Erstellung der Verteilungen innerhalb eines Headway
            if passDistribution == Distribution.beta:
                passengerArrivalTimes = self.generateBeta()

            elif passDistribution == Distribution.uniform:
                passengerArrivalTimes = np.random.rand(numPassengerPerHeadway)*self.hWay
            
            elif passDistribution == Distribution.johnson_sb:
                passengerArrivalTimes = self.generateJohnsonCombined()
            
            passengerArrivalTimes = np.sort(passengerArrivalTimes)
            passengerArrival[len(passengerArrival):] = (cycle*self.hWay)+passengerArrivalTimes
        return passengerArrival

    # Erstellt die Ankunftszeiten der Öffis
    def initVehicleArrival(self):
        
        busArrivals = np.linspace(
            self.hWay, self.runtime, self.numBusses, dtype='float')

        if vehUnpunctuality == Distribution.uniform and u!=0:
            unpunctuality = (np.random.random(size=self.numBusses)-0.5)*2+(headway*u)
        elif vehUnpunctuality == Distribution.normal and u!=0:
            unpunctuality = np.random.normal(loc=0.0, scale=1, size=self.numBusses)+(headway*u) 
            unpunctuality = np.clip(unpunctuality,-headway,+headway)
        elif vehUnpunctuality == Distribution.none:
            unpunctuality = np.zeros(len(busArrivals))
        else: 
            unpunctuality = np.zeros(len(busArrivals))

        unpunctuality = np.clip(unpunctuality,0,self.runtime) #bus kommt zu früh, wartet aber bis zum schedule
        self.unpunctuality = unpunctuality
        busArrivals = busArrivals + unpunctuality  # adding noise to veh. arrival
        # cutting at time=0 and time=runtime
        busArrivals = np.clip(busArrivals, 0, self.runtime)
        busArrivals[len(busArrivals)-1] = self.runtime
        return busArrivals

    #Errechne den Zeitpunkt des nächsten Busses
    def getNextVeh(self, time):

        #Zeit ist weiter als bus -> der nächste Bus ist bereits gekommen -> busCounter +1
        if time >= self.vehArrival[self.busCounter]:
            self.busCounter = self.busCounter+1

        #Der nächste Bus wird aus dem vehArrival Vektor gelesen
        nxtVeh = self.vehArrival[self.busCounter]
        return nxtVeh
    
    #Generiere Betaverteilte Passagiere
    def generateBeta(self):
        #numPassengerPerHeadway 
       
        numBeta = np.int64(np.floor(numPassengerPerHeadway * q))
        r = scipy.stats.beta.rvs(a, b, scale=1, size=numBeta)

        numUni = np.int64(np.ceil(numPassengerPerHeadway * (1 - q)))
        x = np.random.rand(numUni)

        passengerArrivalTimes = np.concatenate((r, x), axis= None)*self.hWay

        return passengerArrivalTimes
    
    #Generiere Johnsonverteilte Passagiere
    def generateJohnsonCombined(self):
        #FROM: Passenger arrival rates at public transport stations Author(s):Lüthi, Marco; Weidmann, Ulrich; Nash, Andrew
        t_hw = self.hWay

        yPDF_vec = np.zeros(numPassengerPerHeadway*10)
        yCDF_vec = np.zeros(numPassengerPerHeadway*10)
 
        x_vec = np.linspace(t_hw/numPassengerPerHeadway*10,t_hw,numPassengerPerHeadway*10)

        i = 0
        for x in x_vec:
            if 0<=x<beta:
                a = (q_john_indep/t_hw)
                b = (q_john*alpha_2*t_hw)/((x+t_hw-beta)*(beta-x)*math.sqrt(2*math.pi))
                c = math.exp(-0.5*pow((alpha_1+alpha_2*math.log((x+t_hw-beta)/(beta-x))),2))
                y = a+b*c
            elif beta<x<=t_hw:
                a = (q_john_indep/t_hw)
                b = (q_john*alpha_2*t_hw)/((x-beta)*(t_hw+beta-x)*math.sqrt(2*math.pi))
                c = math.exp(-0.5*pow((alpha_1+alpha_2*math.log((x-beta)/(t_hw+beta-x))),2))
                y = a+b*c
            else: 
                y = 0
            yPDF_vec[i] = y
            i = i+1

        #Calc CDF
        dx = x_vec[1]-x_vec[0]
        yCDF_vec = np.cumsum(yPDF_vec*dx)
        yCDF_vec = yCDF_vec/yCDF_vec[-1]
        #pos = np.random.randint(low=0,high=len(yCDF_vec),size=10000)
        y_gen = x_vec[yCDF_vec.searchsorted(np.random.rand(int(numPassengerPerHeadway)), 'left')]
        return y_gen

def transportation(env, passNo, station):

    #every passenger has a number (=passNo) and his own arrival time (=arrTime)
    arrTime = station.passArrival[passNo]
    #hold imaginary passenger until his time has come
    yield env.timeout(arrTime)
    #when a passenger arrives he checks the schedule and ...
    nextVeh = station.getNextVeh(env.now)
    #...calculates the waiting time until the next vehicle arrives
    waitingTime = nextVeh - env.now
    
    #log waiting times to array
    station.waitingTime.append(waitingTime)
    return

########################################
vehUnpunctuality        = Distribution.normal
u                       =0.00  #Faktor der Unpünktlichkeit in Prozent vom Headway [0,1] 
unpunctDelay            = []
unpunctWaiting          = []
DataBase                = []
showplots               = True
numPassengerPerHeadway  = 250
numHeadWays             = 3
settings_beta           = [[5, 0.43, 2.85, 0.41],[10, 0.52, 3.39, 0.36],[20, 0.64, 4.57, 0.27],[30, 0.9,  6.52, 0.24],[60, 0.93, 11.2, 0.14]]
settings_johnson        = [[6.33, 0.7, 1-0.7, -1, 1, 0.8000000001],[10, 0.15, 1-0.15, -1.2, 1, 0.2000000001]]
settings_uniform        = [[5],[10],[20],[30],[60]]

#CHANGE passDistribution AND loop seetings... meh!
passDistribution = Distribution.beta
for i in settings_beta:
    
    if passDistribution == Distribution.beta:
        print("┌┐ ┌─┐┌┬┐┌─┐\n├┴┐├┤  │ ├─┤\n└─┘└─┘ ┴ ┴ ┴")
        headway, q, a, b = i[0],i[1],i[2],i[3]
    elif passDistribution == Distribution.johnson_sb:
        print("  ┬┌─┐┌┐┌┌─┐┌─┐┌─┐┌┐┌   ┌─┐┌┐\n  ││ ││││└─┐└─┐│ ││││───└─┐├┴┐\n └┘└─┘┘└┘└─┘└─┘└─┘┘└┘   └─┘└─┘")
        headway, q_john_indep, q_john, alpha_1, alpha_2, beta = i[0],i[1],i[2],i[3],i[4],i[5]
    elif passDistribution == Distribution.uniform:
        print('┬ ┬┌┐┌┬┌─┐┌─┐┬─┐┌┬┐\n│ │││││├┤ │ │├┬┘│││\n└─┘┘└┘┴└  └─┘┴└─┴ ┴')
        headway = i[0]

    runtime     = headway*numHeadWays
    env         = simpy.Environment()
    station     = Station(env, headway, runtime) 

    for passNo in range(len(station.passArrival)):  # jeden passagier durchgehen
        env.process(transportation(env, passNo, station))
    env.run(until=runtime)
    
    print("Headway: {} min.".format(i[0]))
    print("Oh Noooo, our bus is {} min. to late".format(np.mean(station.unpunctuality)/10))
    print("Wartezeit in Minuten: {}".format(np.mean(station.waitingTime)))
    print("Wartezeit in Prozent vom Headway: {}".format(np.mean(station.waitingTime)/headway))
    print(len(station.waitingTime))
    print("--------------------------")
    
    if showplots == True:
        data = station.waitingTime
        scaledWaiting = (data - np.min(data)) / (np.max(data) - np.min(data))
        sns.set_theme(style="darkgrid")
        plt.figure(1)
        sns.set(font_scale = 1.2)
        w = sns.lineplot(data=scaledWaiting,label="{} min. headway".format(headway))

        data =station.passArrival[0:numPassengerPerHeadway-1]
        scaledArrival = (data - np.min(data)) / (np.max(data) - np.min(data))
        plt.figure(2)
        sns.set(font_scale = 1.2)
        p = sns.lineplot(y=scaledArrival,x=np.linspace(0,len(scaledArrival),num=len(scaledArrival)),label="{} min. headway".format(headway), alpha=1)
        DataBase.append([station.waitingTime])

if showplots == True:
    plt.figure(2)
    sns.set_theme(style="darkgrid")
    sns.set(font_scale = 1.2)
    p.set(ylabel="Normalised Headway", xlabel="Number of Passenger")
    if passDistribution == Distribution.beta:
        p.set(title="Beta-Distributed Passenger Arrival Pattern for different Headways")
    elif passDistribution == Distribution.johnson_sb:
        p.set(title="Johnson-SB-Distributed Passenger Arrival Pattern for different Headways")
    elif passDistribution == Distribution.uniform:
        p.set(title="Uniform-Distributed Passenger Arrival Pattern for different Headways")
    p.figure.savefig("PA_{}.png".format(passDistribution.name))

    plt.figure(1)
    sns.set_theme(style="darkgrid")
    #sns.set(font_scale = 1.2)
    # plt.legend(loc='upper right')
    w.set(xlabel="Passenger Number", ylabel="Waiting Time")
    if passDistribution == Distribution.beta:
        w.set(title="Waiting Time for Beta-Distributed Passenger Arrival")
    elif passDistribution == Distribution.johnson_sb:
        w.set(title="Waiting Time for Johnson-Distributed Passenger Arrival")
    elif passDistribution == Distribution.uniform:
        w.set(title="Waiting Time for Uniform-Distributed Passenger Arrival")

    w.figure.savefig("WT_{}.png".format(passDistribution.name))
    plt.show()


x = np.multiply(unpunctDelay,100)
x_john=[0,4.220177507507743, 7.732809066598227, 10.54097651863927, 19.808404536975786, 24.813728049537158, 29.722395847310736]
x_beta=[0,5.2902587883136345, 7.087963446859, 11.18479710908897, 19.962371205955776, 26.486299536356373, 30.593524267291333]
y = np.multiply(unpunctWaiting,100)
y_john=[28.3,32.251015048706414, 35.24760002070181, 37.52439296121476, 45.290069703205155, 48.838609465289274, 52.344647544405575]
y_beta=[28.9,32.09345967516949, 32.82558708551793, 35.160527471997256, 39.7155412097346, 42.826394263188554, 45.01827467995979]

sns.set_theme(style="darkgrid")
sns.set(font_scale = 1.2)
p = sns.lineplot(x=x_john,y=y_john,label="Johnson distributed")
p = sns.lineplot(x=x_beta,y=y_beta,label="Beta distributed")
p.set(xlim=(0, None))
p.set(ylabel="Waiting time [%]",xlabel="Unpunctuality of transportaion system [%]")
p.set(title="Interdependence of unpunctuality and waiting time")
plt.show()