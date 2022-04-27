
import csv
import matplotlib.pyplot as plt

def plott(filename, labelText):
    
    with open(filename) as file:
        data = csv.reader(file)
        next(data) #Removing the top row, because it contains non-numerical info

        print("File opened succesfully.")

        c1 = []
        c2 = []
        c3 = []
        
        last = 0
        for row in data:
            c1.append(float(row[0]))
            c2.append(float(row[1]))
            c3.append(float(row[2]))

    

        print("Ready to plot.")

        plt.ylabel("Spenning [V]")

        plt.xlabel("Tid [s]")
        #plt.xscale("log")

        print()
        print(f"Signal amplitude: {(max(c2)-min(c2))/2}")
        print()

        #plt.ylim(-4, 4)
        #plt.xlim(-0.45,0.45)

        #one = [1 for i in range(len(c1))]

        plt.plot(c1,c2, label=labelText)
        #plt.plot(c2,c2, label="Linje med stigningstall 1")

        

plott("Signal 10M.csv", "y(t)")


plt.legend()
plt.show()