import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

'''
===============================================================================

@params:

@returns:
===============================================================================
'''
def gather_data():
    df = pd.read_csv("ping_pong.dat")
    df.plot(kind='bar', rot=0, subplots=True, marker='o', linestyle='dashed')
    # set the spacing between subplots
    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=0.9, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    #plot(x, y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=12)
    #axes[1].legend(loc=2)
    plt.savefig('output.png')
    #print(data)


'''
===============================================================================

@params:

@returns:
===============================================================================
'''
if __name__ == '__main__':
    gather_data()
