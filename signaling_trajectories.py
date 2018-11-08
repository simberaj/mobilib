
'''Transforms signaling data to smoothed trajectories.'''

import sys

import numpy
import pandas as pd
import geopandas as gpd
import shapely.geometry
import matplotlib.patches
import matplotlib.pyplot as plt

import mobilib.voronoi


SAMPLING = pd.Timedelta('00:01:00')
STD = pd.Timedelta('00:05:00')


def smoothen(array, std_quant):
    return pd.Series(array).rolling(
        int(numpy.ceil(8 * std_quant)),
        min_periods=0,
        center=True,
        win_type='gaussian'
    ).mean(std=std_quant)
    
def trajectory(df, xcol, ycol, sampling, std):
    ts = pd.date_range(df.index.min(), df.index.max(), freq=sampling)
    obs_ind = ts.searchsorted(df.index)
    xs_src = numpy.full(ts.size, numpy.nan)
    xs_src[obs_ind] = df[xcol]
    ys_src = numpy.full(ts.size, numpy.nan)
    ys_src[obs_ind] = df[ycol]
    std_quant = std / sampling
    return smoothen(xs_src, std_quant), smoothen(ys_src, std_quant), ts

if __name__ == '__main__':
    signals = pd.read_csv(sys.argv[1], sep=';')
    signals = signals[signals['phone_nr'] == int(sys.argv[3])]
    signals['pos_time'] = pd.to_datetime(signals['pos_time'])
    timeweights = (1 / signals.groupby('pos_time')['phone_nr'].count()).reset_index().rename(columns={'phone_nr' : 'weight'})
    signals = pd.merge(signals, timeweights, on='pos_time')
    antennas = pd.read_csv(sys.argv[2], sep=';')
    siglocs = pd.merge(signals, antennas, on='cell_name').groupby('pos_time').agg({
        'xcent' : 'mean',
        'ycent' : 'mean',
    })
    xpos, ypos, tpos = trajectory(siglocs, 'xcent', 'ycent', sampling=SAMPLING, std=STD)
    plt.plot(xpos, ypos)
    plt.scatter(antennas.xcent, antennas.ycent, s=9, color='orange')
    plt.gca().set_aspect('equal')
    plt.show()
    pd.DataFrame({'x' : xpos, 'y' : ypos, 't' : tpos}).to_csv(sys.argv[4], sep=';', index=False)