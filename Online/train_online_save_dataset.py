import numpy as np
from DQN_Online import Train_DQN_Online
import random


def train():
    # Predefine discrete delta valuess
    All_Deltas = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    I_Training = []
    J_Training = []
    K_Training = []

    Dev_cord_Training = []

    indx = 0



    # Training
    # Generate random parameters for training. You would save these values and copy them to each training algoithm file
    for c in range(0,10):
        i = random.randint(0, 6) # For the risk region x position
        j = random.randint(0, 5) # For the risk region y position
        print(i)
        print(j)

        Risky_region = np.array([[i,j],[i,j+1],[i,j+2],[i,j+3],[i,j+4],
                                 [i+1,j],[i+1,j+1],[i+1,j+2],[i+1,j+3],[i+1,j+4],
                                 [i+2,j],[i+2,j+1],[i+2,j+2],[i+2,j+3],[i+2,j+4],
                                 [i+3,j],[i+3,j+1],[i+3,j+2],[i+3,j+3],[i+3,j+4]])

        Dev_Coord = np.random.randint(10, size=(10, 2))
        print(Dev_Coord)


        k = random.randint(0, 9) # For the delta value

        print(k)

        DELTA = All_Deltas[k]

        I_Training.append(i)
        J_Training.append(j)
        K_Training.append(k)
        Dev_cord_Training.append(Dev_Coord)

        indx = indx + 1

        Train_DQN_Online(Dev_Coord,Risky_region,DELTA,indx) # Train and save datasets


    print(I_Training)
    print(J_Training)
    print(K_Training)
    print(Dev_cord_Training)


    I_Training = []
    J_Training = []
    K_Training = []

    Dev_cord_Training = []

    # Testing
    # Same as training. Make sure to save the values. Run as many test environments as needed
    for c in range(0,5):
        i = random.randint(0, 6)
        j = random.randint(0, 5)
        print(i)
        print(j)

        Risky_region = np.array([[i,j],[i,j+1],[i,j+2],[i,j+3],[i,j+4],
                                 [i+1,j],[i+1,j+1],[i+1,j+2],[i+1,j+3],[i+1,j+4],
                                 [i+2,j],[i+2,j+1],[i+2,j+2],[i+2,j+3],[i+2,j+4],
                                 [i+3,j],[i+3,j+1],[i+3,j+2],[i+3,j+3],[i+3,j+4]])

        Dev_Coord = np.random.randint(10, size=(10, 2))
        print(Dev_Coord)


        k = random.randint(0, 9)

        print(k)

        DELTA = All_Deltas[k]

        I_Training.append(i)
        J_Training.append(j)
        K_Training.append(k)
        Dev_cord_Training.append(Dev_Coord)

        indx = indx + 1

        Train_DQN_Online(Dev_Coord,Risky_region,DELTA,indx)


    print(I_Training)
    print(J_Training)
    print(K_Training)
    print(Dev_cord_Training)


if __name__ == "__main__":
    train()

    
    