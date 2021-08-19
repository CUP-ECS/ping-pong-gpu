import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

'''
===============================================================================

@params:

@returns:
===============================================================================
'''
def gather_data( mode, direction ):
    df = pd.DataFrame()
    mode_str = ''
    if ( mode == '0' ):
        mode_str = 'direct'
    elif ( mode == '1' ): 
        mode_str = 'cuda'
    elif ( mode == '2' ):
        mode_str = 'copy'
    else:
        print( 'Invalid mode chosen.' )
        return 0
    
    df = pd.read_csv( 'ping_pong_' + mode_str + direction + '.dat' ) 
    #if ( mode == "0" ):
    #    df = pd.read_csv("ping_pong_direct.dat")
    #elif ( mode == "1" ): 
    #    df = pd.read_csv("ping_pong_cuda.dat")
    #elif ( mode == "2" ):
    #    df = pd.read_csv("ping_pong_copy.dat")
    #else:
    #    print( "Invalid mode chosen." )
    #    return 0

    colors = ['r','b','tab:orange']
    
    #df = pd.read_csv("ping_pong.dat")
    axes = df.plot.line( subplots=True, x='max_i', marker='o' )
    #axes[0].set_ylabel( 'seconds' )
    #axes[1].set_ylabel( 'seconds' )
    #axes[2].set_ylabel(   'MiB/s' )

    #axes[0].set_color('r')
    #axes[1].set_color('b')
    #axes[2].set_color('o')

    axes[0].set_ylabel( 'seconds' )
    axes[1].set_ylabel(   'GiB/s' )

    #axes[0].set(ylim=(0,3000))
    #axes[1].set(ylim=(0.00,2))
    #axes[2].set(ylim=(0,0.05e8))

    axes[0].set(ylim=(0.00,2))
    axes[1].set(ylim=(0,1.5e10))

    #axes[0].set(ylim=(0.00,0.08))
    #axes[1].set(ylim=(0,1.5e10))

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
    plt.savefig( 'output_' + mode_str + direction + '_new.png' )
    #if ( mode == "0" ):
    #    plt.savefig('output_direct_new.png')
    #elif ( mode == "1" ): 
    #    plt.savefig('output_cuda_new.png')
    #elif ( mode == "2" ):
    #    plt.savefig('output_copy_new.png')
    #else:
    #    print( "Invalid mode chosen." )
    #print(data)

def main( argv ):
    gather_data( argv[1], argv[2] )

'''
===============================================================================

@params:

@returns:
===============================================================================
'''
if __name__ == '__main__':
    main( sys.argv )
