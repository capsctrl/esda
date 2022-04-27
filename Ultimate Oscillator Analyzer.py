import csv
import matplotlib.pyplot as plt
import numpy as np

def calculate(filename, targetFrequency, frequencyMargin=20):
    
    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        print("[FINDING TARGET FREQUENCY]")
        amplitude = 0
        topFrequency = 0
        for row in data:
            if(float(row[1]) > amplitude) and (float(row[0]) > 500):
                amplitude = float(row[1])
                topFrequency = float(row[0])
 
    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        print("[FINDING SDR]")
        signal = 0
        distortion = 0
        
        for row in data:
            if abs(float(row[0])-topFrequency) < frequencyMargin: #Checks if the frequency on this row is the targeted frequency
                signal += float(row[1])**2
                print(f"Summed signal component: {row[1]}")
            elif float(row[0]) > 1000:
                distortion += float(row[1])**2
            
    print("-------------------------------------------------------------------------------------------")
    print()
    print(f"OSCILLATOR FREQUENCY: {topFrequency}")
    print(f"FREQUENCY DEVIATION: {abs(topFrequency-targetFrequency)}")
    print(f"SIGNAL AMPLITUDE: {amplitude}")
    print(f"SIGNAL EFFECT: {signal}")
    print(f"DISTORTION EFFECT: {distortion}")
    print(f"SIGNAL SDR: {signal/distortion}")
    print(f"SIGNAL SDR [dB]: {10*np.log10(signal/distortion)}")
     
    print("-------------------------------------------------------------------------------------------")
    print()
    print("[DRAWING FREQUENCY SPECTRUM]")

    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        x = []
        y = []
        for row in data:
            x.append(float(row[0]))
            y.append(20*np.log10(float(row[1]))) #Converts it to dB 20*np.log10(
            
        plt.ylabel("Signalstyrke [dB]")
        plt.xlabel("Frekvens [Hz]")

        plt.xscale("log")

        #plt.plot(x, y, label="")
   
        #plt.show()
            

calculate("Spectrum 10M.csv", 5531, 400)
