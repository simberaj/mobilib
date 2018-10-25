
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mobilib.simul


if __name__ == '__main__':
    # numpy.random.seed(42)
    numpy.random.seed(2510)
    netgener = mobilib.simul.NetworkGenerator(
        strength_mean=2,
        strength_stdev=0.5,
        range_mean=1,
        range_stdev=0.5,
        narrowness_mean=0.5,
        strength_sigma=0.75,
    )
    net = netgener.generate((0,0,50,50), 10)
    obs_sys = mobilib.simul.ObservationSystem.create(net, 2500)
    conns = net.connections(obs_sys.observations)
    # ax = plt.gca()
    # obs_sys.plot(ax, net)

    # est = mobilib.simul.MeasureNetworkEstimator()
    # estnet = est.estimate(obs_sys, conns)
    # plt.figure()
    # ax = plt.gca()
    # obs_sys.plot(ax, estnet)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    obs_sys.plot3d(ax, net)
    
    # iest = mobilib.simul.AdjustingNetworkEstimator()
    # iestnet = iest.estimate(obs_sys, conns)
    # plt.figure()
    # ax = plt.gca()
    # obs_sys.plot(ax, iestnet)
    
    plt.show()
