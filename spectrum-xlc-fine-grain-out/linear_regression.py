import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def lin_reg( mode, direction ):
    df = pd.DataFrame()
    mode_str = ''
    if ( mode == '0' ):
        mode_str = 'direct'
    if ( mode == '1' ):
        mode_str = 'cuda'
    if ( mode == '2' ):
        mode_str = 'copy'
    else:
        print( 'Invalid mode, please run with <0,1,2>.' )
        return 0
    df = pd.read_csv( 'ping_pong_' + mode_str + direction + '.dat' )

    x = df[[ "size" ]].to_numpy().reshape((    -1,  1 ))
    y = df[[ "duration" ]].to_numpy().reshape(( 1, -1 ))

    model = LinearRegression().fit( x, y )

    r_sq = model.score( x, y )
    print( f"coefficient of determination: {r_sq}" )
    print( f"intercept: {model.intercept_}"        )
    print( f"slope: {model.coef_}"                 )

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
