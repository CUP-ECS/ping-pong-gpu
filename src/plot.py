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
    #ax = df.plot.gca()
    # df.plot(kind='bar', rot=0, subplots=True)
    axes = df.plot.line(subplots=True, x='max_i') 
    axes[0].set_ylabel('seconds')
    axes[1].set_ylabel('seconds')
    axes[2].set_ylabel('GiB/s')
    #df.plot(kind='line', x='max_i', y='bandwidth', subplots=True)
    #df.plot(kind='line', x='max_i', y='duration', subplots=True)
    #df.plot(kind='line', x='max_i', y='latency', subplots=True)
    # set the spacing between subplots
    #plt.subplots_adjust(left=0.1,
    #                    bottom=0.1, 
    #                    right=0.9, 
    #                    top=0.9, 
    #                    wspace=0.4, 
    #                    hspace=0.4)
    #axes[1].legend(loc=2)
    type(axes)
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
