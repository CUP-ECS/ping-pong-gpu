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
    #df = pd.read_csv("ping_pong_direct.dat")
    #df = pd.read_csv("ping_pong_cuda.dat")
    df = pd.read_csv("ping_pong_copy.dat")
    #df = pd.read_csv("ping_pong.dat")
    axes = df.plot.line(subplots=True, x='max_i', marker='o') 
    axes[0].set_ylabel('seconds')
    axes[1].set_ylabel('seconds')
    axes[2].set_ylabel('GiB/s')

    #df.plot(kind='bar', rot=0, subplots=True, marker='o', linestyle='dashed')
    # set the spacing between subplots
    #plt.subplots_adjust(left=0.1,
    #                    bottom=0.1, 
    #                    right=0.9, 
    #                    top=0.9, 
    #                    wspace=0.4, 
    #                    hspace=0.4)
    #axes[1].legend(loc=2)
    type(axes)
    #plt.savefig('output.png')
    #plt.savefig('output_direct.png')
    #plt.savefig('output_cuda.png')
    plt.savefig('output_copy.png')
    #print(data)


'''
===============================================================================

@params:

@returns:
===============================================================================
'''
if __name__ == '__main__':
    gather_data()
