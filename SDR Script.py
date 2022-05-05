import csv

def calculate(filename, targetFrequency, frequencyMargin=20):
    
    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        signal = 0
        distortion = 0

        amplitude = 0

        iteration = 0
        print("Starting calculation...")
        for row in data:
            if abs(float(row[0])-targetFrequency) < frequencyMargin: #Checks if the frequency on this row is the targeted frequency
                signal += float(row[1])**2
                print(f"Summed signal component: {row[1]}")
            else:
                distortion += float(row[1])**2

            if(float(row[1]) > amplitude):
                amplitude = float(row[1])
            
            iteration += 1
            
    print(f"Total effect of distortion was {distortion}")
    print(f"Total effect of signal was {signal}")
    print(f"SDR for the system is: {signal/distortion}")
    print(f"Amplitude of wanted frequency: {amplitude}")
    
            

calculate("Spectre Output V 10k ohm.csv", 5500)