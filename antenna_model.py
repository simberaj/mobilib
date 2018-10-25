
import numpy
import matplotlib.pyplot as plt

import mobilib.simul


if __name__ == '__main__':
    numpy.random.seed(1711)
    netgener = mobilib.simul.NetworkGenerator(
        strength_mean=2,
        strength_stdev=0.5,
        range_mean=1,
        range_stdev=0.5,
        narrowness_mean=0.5,
        strength_sigma=0.75,
    )
    net = netgener.generate((0,0,20,20), 3)
    obs_sys = mobilib.simul.ObservationSystem.create(net, 500)
    conns = net.connections(obs_sys.observations)
    # ax = plt.gca()
    # obs_sys.plot(ax, net)
    # est = mobilib.simul.MeasureNetworkEstimator()
    # estnet = est.estimate(obs_sys, conns)
    # plt.figure()
    # ax = plt.gca()
    # obs_sys.plot(ax, estnet)
    iest = mobilib.simul.AdjustingNetworkEstimator()
    iestnet = iest.estimate(obs_sys, conns)
    # plt.show()
