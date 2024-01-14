import sys

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_RESOLUTION = pd.to_timedelta('2 hours')
DEFAULT_DAY_SAMPLING = pd.to_timedelta('30 minutes')
DEFAULT_MIN_TIMESPAN = pd.to_timedelta('18 hours')
MIDNIGHT = pd.Timedelta('-30 minutes')
MIDNIGHT_SHIFT = -(MIDNIGHT / pd.Timedelta('1 day')) * 2 * numpy.pi
DAY_NS = 86400000000000
CIRCLE = 2 * numpy.pi
NS_TO_DAYANGLE = DAY_NS / CIRCLE
DAY_TIME = pd.to_timedelta('1 day')


startend = dict(
    start=pd.to_datetime('2016-08-17'),
    end=pd.to_datetime('2016-09-06'),
)


def timepoint_weights(times, resolution=DEFAULT_RESOLUTION, start=None, end=None):
    times = pd.Series(
        [start if start is not None else -resolution],
        dtype=times.dtype
    ).append([
            times,
            pd.Series(
                [
                    end if end is not None
                    else times.iat[len(times)-1] + resolution
                ],
                dtype=times.dtype
            )
        ],
        ignore_index=True
    )
    return pd.to_timedelta(
        times.astype(numpy.int64).diff().rolling(2).mean()
    ).clip_upper(resolution).iloc[2:].values


def circmean(ts):
    times = (ts.astype(numpy.int64) % DAY_NS) / NS_TO_DAYANGLE
    return pd.Timestamp((numpy.arctan2(numpy.sin(times).sum(), numpy.cos(times).sum()) * NS_TO_DAYANGLE) % DAY_NS)


def dayprofile(ts, weights, resolution=DEFAULT_RESOLUTION, sampling=DEFAULT_DAY_SAMPLING):
    sd = resolution / DAY_TIME
    n = int(numpy.ceil(DAY_TIME / sampling))
    times = numpy.round((ts.astype(numpy.int64) % DAY_NS) / DAY_NS * n).astype(int)
    weights_norm = weights / resolution
    dens = numpy.zeros(n)
    kernel = scipy.stats.norm(loc=.5, scale=sd).pdf(numpy.linspace(0,1,n))
    initconst = n // 2
    for time, wnorm in zip(times, weights_norm):
        dens += numpy.roll(kernel, time - initconst) * wnorm
    return dens


def night_index(profile):
    return (profile / profile.sum()).dot(numpy.cos(numpy.linspace(
        -MIDNIGHT_SHIFT, 2 * numpy.pi - MIDNIGHT_SHIFT, profile.size
    )))


def cell_agg(subdf):
    profile = dayprofile(subdf.timestamp, subdf.timeweight)
    timespan = subdf.timestamp.max() - subdf.timestamp.min()
    cell_circmean = circmean(subdf.timestamp)
    night_i = night_index(profile)
    twsum = subdf.timeweight.sum()
    if timespan > DEFAULT_MIN_TIMESPAN:
        plt.plot(numpy.linspace(0, 24, profile.size), profile)
        plt.title('TS {span}, TW {twsum}, {n} obs., NI {ni:.4f}'.format(
            span=timespan,
            n=len(subdf.index),
            cm=cell_circmean.time(),
            ni=night_i,
            twsum=twsum,
        ))
        plt.show()
    d = {
        'profile_max' : pd.Timestamp(profile.argmax() / profile.size * DAY_NS),
        'circmean' : cell_circmean,
        'profile_weight' : profile.sum(),
        'timespan' : timespan,
        'timeweight' : twsum,
        'night_index' : night_i,
        'is_home' : night_i > 0,
        'observations' : len(subdf.index),
        'profile' : profile,
    }
    return pd.Series(d, index=d.keys())


def neighbourhood_table(neighdf):
    neightable = {}
    for cid1, cid2 in neighdf.itertuples(index=False, name=None):
        neightable.setdefault(cid1, set()).add(cid2)
        neightable.setdefault(cid2, set()).add(cid1)
    return neightable


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], sep=';')
    neighdf = neighbourhood_table(pd.read_csv(sys.argv[2], sep=';'))
    df.timestamp = pd.to_datetime(df.timestamp)
    for uid, udf in df.groupby('uid'):
        udf = udf.copy()
        udf['timeweight'] = timepoint_weights(udf.timestamp, **startend)
        celldf = udf.groupby('cid').apply(cell_agg)
        for cid, celldesc in celldf.sort_values(by='timeweight', ascending=False).iterrows():

        raise RuntimeError