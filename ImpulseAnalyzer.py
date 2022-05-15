import csv
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq

#Copyright Mathias Dam 2022

pad = 20

#This function opens a csv file exported from WaveLabs, reads it and returns the info contained in it as three
#python lists, one with time values, one with channel 1 values and one with channel 2 values.
def openFile(filename):
    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        t = []
        c1 = []
        c2 = []
        for row in data:
            t.append(float(row[0]))
            c1.append(float(row[1]))
            c2.append(float(row[2]))

        plt.ylabel("y(t)")
        plt.xlabel("z(t)")

        return t,c1,c2

def fourier_transform(values, timesteps, n, m, freq_max):
    T = timesteps[-1]

    g = lambda t, freq: np.exp(-1j*2*np.pi*freq*t)
    
    freqs = np.linspace(0, freq_max, n)
    l = np.linspace(0, len(values) - 1, m).astype(int)
    final = []
    print("Calculating fourier transform:")
    for freq in tqdm(freqs):
        total = 0
        for i in range(len(values)):
            total += values[i] * g(timesteps[i], freq)
            
        final.append(total / len(values))
    
    return freqs, final

def deleteBeforeZero(t,c1):
    for i in range(len(c1)):
        if t[i] >= 0:
            startIndex = i
            break
    return t[startIndex:], c1[startIndex:]

#Advanced math aw yea, these are derived in the paper
def teorethicalLenght(R, C, Vmax, Vt):
    return R*C*np.log(Vmax/Vt)

def theoreticalImpulseResponse(R, C, L, t):
    a = R/(2*L)
    om = np.sqrt((1/(C*L)) - (R/(2*L))**2)
    return 2*a*np.e**(-a*t)*(np.cos(om*t)-(1/om)*np.sin(om*t))

def tFrequencyResponse(f, R, C, L):
    w= 2*np.pi*f
    return (R)/(1j*w*L + R + 1/(1j*w*C))

#This function is used for analysing the impulses created
def analyzeImpulse(t,c1,treshold, R, C, Vmax, Vt):
    foundStart = False
    
    print(teorethicalLenght(R,C,Vmax,Vt))
    print(max(c1)) #Amplitude in case the impulse is scuffed
    #First, we find the beginning, end and amplitude of the impulse
    for i in range(len(c1)):
        if c1[i] > treshold and not foundStart:
            startTime = t[i]
            amplitude = c1[i+pad]
            startIndex = i
            foundStart=True

        if foundStart and i > startIndex+pad and c1[i] < amplitude/2:
            endTime = t[i]
            print(endTime)
            endIndex = i
            break
    
    #Now, time to calculate SDR
    distortion = 0
    signal = 0
    y=[]
    z=[]
    x=[]
    for i in range(endIndex-startIndex+110): #Takes 10 datapoints before the impulse and 100 after, calculates SDR
        if i < 10:
            distortion += abs(c1[startIndex+i-10])
            z.append(0)
        elif t[startIndex + i - 10] < endTime:
            signal += amplitude
            distortion += abs(amplitude - c1[startIndex+i-10])
            z.append(amplitude)
        else: 
            distortion += abs(c1[startIndex+i-10])
            z.append(0)
        
        x.append(t[startIndex+i-10])
        y.append(c1[startIndex+i-10])
        
    print()
    print(f"Impulse length: {endTime-startTime} s")
    print(f"Theoretical lenght: {teorethicalLenght(R,C,Vmax,Vt)} s")
    print(f"Lenght Difference: {(endTime-startTime)-teorethicalLenght(R,C,Vmax,Vt)} s")
    print(f"Impulse amplitude: {amplitude} V")
    print(f"SDR [dB]: {20*np.log10(signal/distortion)} dB")
    print()

    #Then we plot the data that was used to calculate all the stuff, just to make sure that we included the whole pulse
    #and that nothing wierd is giong on 
    plt.plot(x,z)
    plt.plot(x,y)
    plt.show()


#This one is used to analyse the impulse response
def analyzeImpulseResponse(te,ce1, R, C, L): #Impulse response has to start in t=0
    #The first thing we do is to delete all values before t = 0 (those values are absolute losers!)
    
    #First, we will make a list with theoretical graph.
    ce2 = []
    t0 = te[0]
    for i in range(len(ce1)):
        ce2.append(theoreticalImpulseResponse(R,C,L,te[i]-t0-18.42*10**-6))
    
    #Then, we need to normalize them so they have the same amplitude..
    normalizeFactor = sum([abs(i) for i in ce1])/sum([abs(i) for i in ce2])
    for i in range(len(ce2)):
        ce2[i]*=normalizeFactor
    
    

    #Then its finally time to calculate the SDR
    signal = 0
    distortion = 0
    for i in range(len(te)):
        signal += abs(ce2[i])
        distortion += abs(ce1[i]-ce2[i])
    
    print()
    print(f"SDR: {20*np.log10(signal/distortion)} dB")
    print()

    plt.ylabel("Spenning [dB V]")
    plt.xlabel("Tid [s]")

    #Then we plot to see that all looks right:
    plt.plot(te,ce1, label="Målingsverdi")
    plt.plot(te,ce2, label="Teoretisk verdi")
    plt.legend()
    plt.show()

def impulseSpectre(t, c1, R, C, L):
    # Number of sample points
    N = len(t)
    frange = 100
    # sample spacing (time difference from one sample to the next)
    T = t[len(t)-1]/len(t)
    x = t
    y = c1
    yf = fft(y)
    xf = fftfreq(N, T)[:N//frange]
    import matplotlib.pyplot as plt

    plt.ylabel("Spenning [dB V]")
    plt.xlabel("Frekvens [Hz]")

    #plt.xscale("log")
    plt.yscale("log")

    plt.plot(xf, 2.0/N * np.abs(yf[0:N//frange]))
    plt.grid()
    plt.show()   

def plotNetwork(filename):
    t,c1,c2 = openFile(filename)

    print("Resonant frequency of: Network " + f" is {findResFreq(t, c2)} Hz")

    plt.plot(t, c2, label="'Network' analyse")

def findResFreq(t, c1):
    resfreq = 0
    amp = 0
    for i in range(len(t)):
        if c1[i] > amp:
            amp = c1[i]
            resfreq = t[i]
    
    return resfreq

def plotSpectres(filename, label):
    t,c1,c2 = openFile(filename)

    #analyzeImpulse(t, c1, 2, 1000, 47*10**-9, 5, 2.2)

    #Processing em
    tp, c2p = deleteBeforeZero(t, c2)
    tp, c1p = deleteBeforeZero(t, c1)
    #analyzeImpulseResponse(tp, c2p, 1000, 3.3*10**-9, 0.240)

    #impulseSpectre(tp, c2p, 1000, 3.3*10**-9, 0.240)

    freqs, final = fourier_transform(c2p, tp, 500, 3000, 20000)
    freqs2, final2 = fourier_transform(c1p, tp, 500, 3000, 20000)

    #dividing for realistic gain
    for i in range(len(freqs)):
        final[i] = (final[i]/final2[i])


    print("Resonant frequency of: " + filename + f" is {findResFreq(freqs, final)} Hz")

    

    #plt.xscale("log")
    #plt.yscale("log")

    plt.plot(freqs, final, label=label)
    #plt.ylim(0, 300)



plotSpectres("47F_BigImpulse.csv", "Konfigurasjon B")
#plotSpectres("3p3F_MediumImpulse.csv", "Konfigurasjon A")


plotNetwork("47F_Network.csv")

#plt.xlim(0,12000)

plt.ylabel("Forsterkning")
plt.xlabel("Frekvens [Hz]")

plt.legend()
plt.show()


















# xs = []
# ys = []
# for i in tqdm(range(1,10000)):
#     xs.append(i)
#     ys.append(tFrequencyResponse(i,1000,3.3*10**-9, 0.240)/800)

# plt.plot(xs, ys)
# plt.show()














































































#Hello :)