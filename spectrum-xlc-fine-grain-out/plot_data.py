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
    mode_str1 = ''
    mode_str2 = ''
    mode_str3 = ''
    mode_str  = ''
    mode_strs = []
    if ( mode == '0' ):
        mode_str1 = 'direct'
        mode_str  = 'direct'
    elif ( mode == '1' ): 
        mode_str2 = 'cuda'
        mode_str  = 'cuda'
    elif ( mode == '2' ):
        mode_str3 = 'copy'
        mode_str  = 'copy'
    elif ( mode == '3' ):
        mode_str1 = 'direct'
        mode_str2 = 'cuda'
        mode_str3 = 'copy'
        mode_str  = 'all'
    else:
        print ( 'Invalid mode specified.' )
        return 0
    #mode_strs.append( mode_str1 )
    mode_strs.append( mode_str2 )
    #mode_strs.append( mode_str3 )
    #df1 = pd.read_csv( 'ping_pong_' + mode_str1 + direction + '.dat' ) 
    df2 = pd.read_csv( 'ping_pong_' + mode_str2 + direction + '.dat' ) 
    #df3 = pd.read_csv( 'ping_pong_' + mode_str3 + direction + '.dat' ) 
    #if ( mode == "0" ):
    #    df = pd.read_csv("ping_pong_direct.dat")
    #elif ( mode == "1" ): 
    #    df = pd.read_csv("ping_pong_cuda.dat")
    #elif ( mode == "2" ):
    #    df = pd.read_csv("ping_pong_copy.dat")
    #else:
    #    print( "Invalid mode chosen." )
    #    return 0

    #colors = ['r','b','tab:orange']
    colors = ['tab:orange']
    fig, ax = plt.subplots()
    #fig, ax = plt.subplots()
    #df = pd.read_csv("ping_pong.dat")
    idx = 0
    #plt.title('Bandwidth Direct, CUDA, and Copy; x Direction')
    for frame in [df2]:
    #for frame in [df1, df2, df3]:
        plt.plot(frame['size'], frame['duration'] / 10**6, marker='o', label=mode_strs[idx])
        plt.legend(loc='upper left')
        idx += 1
    #plt.ylim([1,10**4.5])
    #plt.ylim([10**-7,10**-6])
    #axes1 = df1.plot.line( x='max_i', marker='o', ax=ax )
    #axes2 = df2.plot.line( x='max_i', marker='o', ax=ax )
    #axes3 = df3.plot.line( x='max_i', marker='o', ax=ax )
    #axes1 = df1.plot.line( subplots=True, x='max_i', marker='o' )
    #axes2 = df2.plot.line( subplots=True, x='max_i', marker='o' )
    #axes3 = df3.plot.line( subplots=True, x='max_i', marker='o' )
    #axes[0].set_ylabel( 'seconds' )
    #axes[1].set_ylabel( 'seconds' )
    #axes[2].set_ylabel(   'MiB/s' )

    #axes[0].set_color('r')
    #axes[1].set_color('b')
    #axes[2].set_color('o')

    #ax.set_ylabel( 'Bandwidth (MiB/s)' )
    ax.set_ylabel( 'Duration' )
    ax.set_xlabel( 'Message Size' )
    #axes1[0].set_ylabel( 'seconds' )
    #axes1[1].set_ylabel(   'GiB/s' )


    #axes[0].set(ylim=(0,3000))
    #axes[1].set(ylim=(0.00,2))
    #axes[2].set(ylim=(0,0.05e8))

    #ax[0].set(ylim=(0.00,2))
    #ax[1].set(ylim=(0,1.5e10))

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
    #type(axes1)
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
