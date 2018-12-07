
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import mobilib.antenna

if __name__ == '__main__':
    # numpy.random.seed(42)
    numpy.random.seed(2510)
    # netgener = mobilib.antenna.FixedAngleNetworkGenerator(
    netgener = mobilib.antenna.VariableAngleNetworkGenerator(
        strength_mean=2,
        strength_stdev=0.5,
        range_mean=1,
        range_stdev=0.5,
        narrowness_mean=0.5,
        # strength_sigma=0.75,
        # strength_sigma=0.1,
        strength_sigma=0,
        # grouping_factor=2.5,
    )
    net = netgener.generate((0,0,50,50), 10)
    # net = netgener.generate((0,0,50,50), 15)
    print(net)
    obs_sys = mobilib.antenna.ObservationSystem.create(
        net, 500,
        weighter_class=mobilib.antenna.VoronoiAreaObservationWeighter
    )
    print(obs_sys)
    conns = net.connections(obs_sys.observations, p=1)
    print(conns)
    ax = plt.gca()
    obs_sys.plot(ax, net)

    # est = mobilib.antenna.MeasureNetworkEstimator()
    # # est = mobilib.antenna.AdjustingNetworkEstimator()
    # estnet = est.estimate(obs_sys, network=network)
    # plt.figure()
    # ax = plt.gca()
    # obs_sys.plot(ax, estnet)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    obs_sys.plot3d(ax, net)
    
    # iest = mobilib.antenna.AdjustingNetworkEstimator()
    # iestnet = iest.estimate(obs_sys, conns)
    # plt.figure()
    # ax = plt.gca()
    # obs_sys.plot(ax, iestnet)
    
    plt.show()
