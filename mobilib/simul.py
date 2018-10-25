
import numpy
import scipy.stats
import shapely.geometry
import shapely.geometry.base
import shapely.prepared

from . import vector
from . import voronoi

class Mast:
    def __init__(self, location):
        self.location = numpy.array(location)

    def dirvectors(self, locations):
        return locations - self.location


class Antenna:
    PARAM_FORMAT = '{0.strength:.2f}/{0.range:.2f}@{0.principal_angle_deg:.0f}w{0.narrowness:.2f}'
    ALL_FORMAT = (
        '<Antenna({0.mast.location[0]:.2f},{0.mast.location[1]:.2f};'
        + PARAM_FORMAT + ')'
    )

    def __init__(self, mast, strength, range, principal_angle, narrowness, strength_sigma=0):
        self.mast = mast
        self.strength = strength
        self.strength_sigma = strength_sigma
        self.range = range
        self.principal_angle = principal_angle
        self.narrowness = narrowness
        self._update_distros()

    @property
    def location(self):
        return self.mast.location

    def _update_distros(self):
        self._strength = scipy.stats.norm(loc=self.strength, scale=self.strength_sigma)
        self._distance = scipy.stats.norm(scale=self.range)
        self._angle = scipy.stats.vonmises(loc=self.principal_angle, kappa=self.narrowness)

    def strengths(self, locations):
        dirvec = self.mast.dirvectors(locations)
        return self.strengths_from_distangles(vector.length(dirvec), vector.angle(dirvec))

    def strengths_from_distangles(self, distances, angles):
        return (
            self._strength.rvs(len(distances))
            * self._distance.pdf(distances)
            * self._angle.pdf(angles)
        )

    @property
    def principal_angle_deg(self):
        return float(numpy.degrees(self.principal_angle))

    def plot_annotation(self, ax):
        label = self.PARAM_FORMAT.format(self)
        ax.annotate(label, xy=self.location, xytext=self.location)
    
    def __repr__(self):
        return self.ALL_FORMAT.format(self)


def generate_locations(extent, n):
    if isinstance(extent, shapely.geometry.base.BaseGeometry):
        extent = extent
    else:
        extent = shapely.geometry.box(*extent)
    extent_prep = shapely.prepared.prep(extent)
    xlocgen, ylocgen = _create_locgens(*extent.bounds)
    locs = []
    while len(locs) < n:
        location = numpy.hstack([xlocgen.rvs(1), ylocgen.rvs(1)])
        if extent_prep.contains(shapely.geometry.Point(*location)):
            locs.append(location)
    return numpy.array(locs)


def _create_locgens(xmin, ymin, xmax, ymax):
    return (
        scipy.stats.uniform(loc=xmin, scale=xmax-xmin),
        scipy.stats.uniform(loc=ymin, scale=ymax-ymin),
    )


class NetworkGenerator:
    def __init__(self, strength_mean, strength_stdev, range_mean, range_stdev, narrowness_mean, strength_sigma=0):
        self.strength_mean = strength_mean
        self.strength_stdev = strength_stdev
        self.strength_sigma = strength_sigma
        self.range_mean = range_mean
        self.range_stdev = range_stdev
        self.narrowness_mean = narrowness_mean
        self._generators = {
            'strength' : scipy.stats.norm(loc=self.strength_mean, scale=self.strength_stdev),
            'range' : scipy.stats.norm(loc=self.range_mean, scale=self.range_stdev),
            'principal_angle' : scipy.stats.uniform(scale=2 * numpy.pi),
            'narrowness' : scipy.stats.expon(scale=self.narrowness_mean),
        }

    def generate(self, extent, n_antennas):
        return AntennaNetwork(extent, [Antenna(
                mast=Mast(location),
                strength_sigma=self.strength_sigma,
                **{
                    key : float(gener.rvs(1))
                    for key, gener in self._generators.items()
                }
            )
            for location in generate_locations(extent, n_antennas)
        ])


class AntennaNetwork:
    def __init__(self, extent, antennas):
        self.extent = extent
        self.antennas = antennas
        self._antenna_locs = numpy.stack([antenna.location for antenna in antennas])

    def connections(self, locations):
        return self.strengths_to_connections(numpy.stack([
            antenna.strengths(locations) for antenna in self.antennas
        ]))

    def connections_from_distangles(self, distances, angles):
        return self.strengths_to_connections(numpy.stack([
            antenna.strengths_from_distangles(distances, angles)
            for antenna in self.antennas
        ]))

    @staticmethod
    def strengths_to_connections(strengths):
        return strengths.argmax(axis=0)

    def plot(self, ax):
        ax.scatter(
            self._antenna_locs[:,0],
            self._antenna_locs[:,1],
            c=['C' + str(i) for i in range(self.n_antennas)],
            s=64,
        )
        for antenna in self.antennas:
            antenna.plot_annotation(ax)

    @property
    def n_antennas(self):
        return len(self.antennas)

    def get_masts(self):
        return [antenna.mast for antenna in self.antennas]

    def __repr__(self):
        return repr(self.antennas)


class ObservationSystem:
    def __init__(self, extent, masts, observations):
        self.extent = extent
        self.masts = masts
        self._mast_locs = numpy.stack([mast.location for mast in self.masts])
        self.observations = observations
        self.dirvectors = numpy.stack([
            mast.dirvectors(self.observations)
            for mast in self.masts
        ])
        self.distances = vector.length(self.dirvectors)
        self.angles = vector.angle(self.dirvectors)
        self.unitvectors = self.dirvectors / self.distances[:,:,numpy.newaxis]
        self.cells = list(voronoi.cells(self.observations, extent))
        self.weights = numpy.array([cell.area for cell in self.cells])

    def connections(self, network):
        return network.connections(self.observations)
    
    def connection_matrix(self, network, **kwargs):
        return self.connections_to_matrix(self.connections(network), **kwargs)
        
    def connections_to_matrix(self, connections, weighted=False):
        connmatrix = (
            numpy.arange(self.n_masts)[:,numpy.newaxis]
            == connections[numpy.newaxis,:]
        )
        if weighted:
            return connmatrix * self.weights[numpy.newaxis,:]
        else:
            return connmatrix
        
    @property
    def n_masts(self):
        return len(self.masts)
        
    @classmethod
    def create(cls, network, n_observations):
        return cls(
            network.extent,
            network.get_masts(),
            generate_locations(network.extent, n_observations),
        )

    def plot(self, ax, network=None):
        if network is None:
            colors = 'b.'
            cellcolors = itertools.repeat('#bbbbbb')
        else:
            connections = network.connections(self.observations)
            colors = ['C' + str(i) for i in connections]
            cellcolors = colors
        for cell, color in zip(self.cells, cellcolors):
            ax.plot(*cell.exterior.xy, color=color, lw=0.25)
        ax.scatter(
            self.observations[:,0], self.observations[:,1],
            c=colors, s=5 * self.weights
        )
        if network is not None:
            network.plot(ax)

            
class AntennaNetworkEstimator:
    def estimate(self, system, connections):
        raise NotImplementedError
        
    def _network_from_params(self, system, **kwargs):
        return AntennaNetwork(system.extent, [Antenna(
                mast=mast,
                **{
                    key : values[i]
                    for key, values in kwargs.items()
                }
            )
            for i, mast in enumerate(system.masts)
        ])
    
class MeasureNetworkEstimator(AntennaNetworkEstimator):
    def estimate(self, system, connections):
        connmatrix = system.connections_to_matrix(connections, weighted=True)
        sincossums = (
            (system.unitvectors * connmatrix[:,:,numpy.newaxis]).sum(axis=1)
            / connmatrix.sum(axis=1)[:,numpy.newaxis]
        )
        # strengths assumed equal
        strengths = numpy.ones(system.n_masts)
        # range is mean distance of connected point from antenna
        ranges = (system.distances * connmatrix).sum(axis=1) / connmatrix.sum(axis=1)
        # principal angle is mean angle of connected point from antenna
        princ_angles = vector.angle(sincossums)
        narrownesses = self._estimate_kappa(vector.length(sincossums))
        return self._network_from_params(system,
            strength=strengths,
            range=ranges,
            principal_angle=princ_angles,
            narrowness=narrownesses
        )
        
    def _estimate_kappa(self, rbar, n_iter=3):
        rbarsq = rbar ** 2 
        kappa = rbar * (2 - rbarsq) / (1 - rbarsq)
        for i in range(n_iter):
            apkappa = scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)
            kappa -= (apkappa - rbar) / (1 - apkappa ** 2 - apkappa / kappa)
        return kappa
        
        
class AdjustingNetworkEstimator(AntennaNetworkEstimator):
    def estimate(self, system, connections):
        connmatrix = system.connections_to_matrix(connections, weighted=True)
        inner_estimator = MeasureNetworkEstimator()
        estnet = inner_estimator.estimate(system, connections)
        est_connections = estnet.connections(system.observations)
        est_connmatrix = system.connections_to_matrix(est_connections, weighted=True)
        prev_fit = -1
        fit = numpy.sqrt(connmatrix * est_connmatrix).sum() / connmatrix.sum()
        # while (fit - prev_fit) > 1e-3:
        totweights = est_connmatrix.sum(axis=1)
        print(totweights)
            
            # print(fit)