
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


class DirectedMast(Mast):
    def __init__(self, location, angle_deg=None):
        super().__init__(location)
        self.angle_deg = angle_deg
        self.angle = numpy.radians(self.angle_deg) if angle_deg is not None else None

    def get_angle(self):
        return self.angle


class Antenna:
    PARAM_FORMAT = '{0.strength:.2f}/{0.range:.2f}@{0.principal_angle_deg:.0f}w{0.narrowness:.2f}'
    ALL_FORMAT = (
        '<Antenna({0.mast.location[0]:.2f},{0.mast.location[1]:.2f};'
        + PARAM_FORMAT + ')'
    )
    EPSILON_RANGE = 1e-7

    def __init__(self, mast, strength, range, narrowness, strength_sigma=0):
        self.mast = mast
        self.strength = float(strength)
        self.strength_sigma = strength_sigma
        self.range = float(range)
        if abs(self.range) < self.EPSILON_RANGE:
            self.range = self.EPSILON_RANGE
        self.narrowness = float(narrowness)
        self._update_distros()

    @property
    def location(self):
        return self.mast.location

    @property
    def principal_angle_deg(self):
        return (
            float(numpy.degrees(self.principal_angle))
            if self.principal_angle is not None
            else None
        )

    def _update_distros(self):
        self._strength = scipy.stats.norm(loc=self.strength, scale=self.strength_sigma)
        # self._distance = scipy.stats.norm(scale=self.range)
        self._distance = scipy.stats.cauchy(scale=self.range)
        if self.principal_angle is None or self.narrowness == 0:
            self._angle = scipy.stats.uniform(scale=2 * numpy.pi)
        else:
            self._angle = scipy.stats.vonmises(
                loc=self.principal_angle,
                kappa=self.narrowness
            )

    def strengths(self, locations):
        dirvec = self.mast.dirvectors(locations)
        return self.strengths_from_distangles(vector.length(dirvec), vector.angle(dirvec))

    def strengths_from_distangles(self, distances, angles):
        return (
            self._strength.rvs(len(distances))
            * self._distance.pdf(distances)
            * self._angle.pdf(angles)
        )

    def plot_annotation(self, ax):
        label = self.PARAM_FORMAT.format(self)
        ax.annotate(label, xy=self.location, xytext=self.location)

    def __repr__(self):
        return self.ALL_FORMAT.format(self)


class VariableAngleAntenna(Antenna):
    VAR_NAMES = ['strength', 'range', 'narrowness', 'principal_angle']

    def __init__(self, *args, principal_angle=0, **kwargs):
        self.principal_angle = principal_angle
        super().__init__(*args, **kwargs)


class FixedAngleAntenna(Antenna):
    VAR_NAMES = ['strength', 'range', 'narrowness']

    @property
    def principal_angle(self):
        return self.mast.get_angle()


def random_locations(extent, n):
    return filter_by_extent(random_locations_infinite(extent), extent, n)


def rectgrid_locations(extent, grid_size):
    return filter_by_extent(rectgrid_locations_all(extent), extent)


def filter_by_extent(generator, extent, n=numpy.inf):
    if not isinstance(extent, shapely.geometry.base.BaseGeometry):
        extent = shapely.geometry.box(*extent)
    extent_prep = shapely.prepared.prep(extent)
    locs = []
    for loc in generator:
        if extent_prep.contains(shapely.geometry.Point(*loc)):
            locs.append(loc)
            if len(locs) == n:
                break
    return numpy.array(locs)


def random_locations_infinite(extent):
    xmin, ymin, xmax, ymax = to_bounds(extent)
    xlocgen = scipy.stats.uniform(loc=xmin, scale=xmax-xmin)
    ylocgen = scipy.stats.uniform(loc=ymin, scale=ymax-ymin)
    while True:
        yield numpy.hstack([xlocgen.rvs(1), ylocgen.rvs(1)])


def rectgrid_locations_all(extent, grid_size):
    xmin, ymin, xmax, ymax = to_bounds(extent)
    xeq = (xmax - xmin - ((xmax - xmin) // grid_size) * grid_size) / 2
    yeq = (ymax - ymin - ((ymax - ymin) // grid_size) * grid_size) / 2
    xs = numpy.arange(xmin + xeq, xmax, grid_size)
    ys = numpy.arange(ymin + yeq, ymax, grid_size)
    for x in xs:
        for y in ys:
            yield numpy.array([x,y])

def to_bounds(extent):
    if isinstance(extent, shapely.geometry.base.BaseGeometry):
        return extent.bounds
    else:
        return extent



class NetworkGenerator:
    def __init__(self, strength_mean, strength_stdev, range_mean, range_stdev, narrowness_mean, strength_sigma=0):
        self.strength_mean = strength_mean
        self.strength_stdev = strength_stdev
        self.strength_sigma = strength_sigma
        self.range_mean = range_mean
        self.range_stdev = range_stdev
        self.narrowness_mean = narrowness_mean
        self._strength_gen = scipy.stats.norm(loc=self.strength_mean, scale=self.strength_stdev)
        self._range_gen = scipy.stats.norm(loc=self.range_mean, scale=self.range_stdev)
        self._narrowness_gen = scipy.stats.expon(scale=self.narrowness_mean)
        self._angle_gen = scipy.stats.uniform(scale=2 * numpy.pi)

    def generate(self, extent, n_antennas):
        return AntennaNetwork(extent, list(self._generate_antennas(extent, n_antennas)))


class VariableAngleNetworkGenerator(NetworkGenerator):
    def _generate_antennas(self, extent, n_antennas):
        for location in random_locations(extent, n_antennas):
            yield VariableAngleAntenna(
                mast=Mast(location),
                strength_sigma=self.strength_sigma,
                strength=self._strength_gen.rvs(1),
                range=self._range_gen.rvs(1),
                narrowness=self._narrowness_gen.rvs(1),
                principal_angle=self._angle_gen.rvs(1),
            )


class FixedAngleNetworkGenerator(NetworkGenerator):
    def __init__(self, *args, grouping_factor=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.grouping_factor = grouping_factor

    def _generate_antennas(self, extent, n_antennas):
        n_masts = int(numpy.ceil(n_antennas / self.grouping_factor))
        antenna_masts = (numpy.random.rand(n_antennas) * n_masts).astype(int)
        mast_antenna_counts = [(antenna_masts == i).sum() for i in range(n_masts)]
        mast_locs = random_locations(extent, n_masts)
        for mast_i in antenna_masts:
            if mast_antenna_counts[mast_i] == 1:
                narrowness = 0.0
                angle = 0.0
            else:
                narrowness = self._narrowness_gen.rvs(1)
                angle = self._angle_gen.rvs(1)
            yield FixedAngleAntenna(
                mast=DirectedMast(mast_locs[mast_i], angle),
                strength_sigma=self.strength_sigma,
                strength=self._strength_gen.rvs(1),
                range=self._range_gen.rvs(1),
                narrowness=narrowness
            )


class AntennaNetwork:
    def __init__(self, extent, antennas):
        self.extent = extent
        self.antennas = antennas
        self._antenna_locs = numpy.stack([antenna.location for antenna in antennas])

    def strengths(self, locations):
        return numpy.stack([
            antenna.strengths(locations) for antenna in self.antennas
        ])

    def strengths_from_distangles(self, distances, angles):
        return numpy.stack([
            antenna.strengths_from_distangles(dists, angs)
            for dists, angs, antenna in zip(distances, angles, self.antennas)
        ])

    def connections(self, locations, p=0):
        return self.strengths_to_connections(self.strengths(locations), p=p)

    def connections_from_distangles(self, distances, angles, p=0):
        return self.strengths_to_connections(
            self.strengths_from_distangles(distances, angles),
            p=p
        )

    @staticmethod
    def strengths_to_connections(strengths, p=0):
        if p == 0:
            return strengths.argmax(axis=0)
        else:
            return (strengths * p) >= strengths.max(axis=0)

    def plot(self, ax):
        ax.scatter(
            self._antenna_locs[:,0],
            self._antenna_locs[:,1],
            c=['C' + str(i % 10) for i in range(self.n_antennas)],
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
        return '<AntennaNetwork' + repr(self.antennas) + '>'


class ObservationSystem:
    PARAM_NAMES = {
        Mast : ['strength', 'range', 'principal_angle', 'narrowness'],
        DirectedMast : ['strength', 'range', 'narrowness'],
    }
    ANTENNA_TYPES = {
        Mast : VariableAngleAntenna,
        DirectedMast : FixedAngleAntenna,
    }

    def __init__(self, extent, masts, observations, weighter_class):
        self.extent = extent
        self.masts = masts
        self.mast_type = type(self.masts[0])
        # print(set(tuple(mast.location) for mast in self.masts))
        self.observations = numpy.concatenate((
            observations,
            numpy.array(list(set(tuple(mast.location) for mast in self.masts)))
        ))
        self.dirvectors = numpy.stack([
            mast.dirvectors(self.observations)
            for mast in self.masts
        ])
        self.distances = vector.length(self.dirvectors)
        self.angles = vector.angle(self.dirvectors)
        self.unitvectors = self.dirvectors / numpy.where(self.distances == 0, 1, self.distances)[:,:,numpy.newaxis]
        self.weighter = weighter_class(self.observations, self.extent)

    @property
    def weights(self):
        return self.weighter.weights

    def get_param_names(self):
        return self.PARAM_NAMES[self.mast_type]
    
    def get_antenna_class(self):
        return self.ANTENNA_TYPES[self.mast_type]

    def strengths(self, network):
        return network.strengths_from_distangles(self.distances, self.angles)

    def connections(self, network, p=0):
        return network.connections_from_distangles(self.distances, self.angles, p=p)

    def connection_matrix(self, network, p=0, **kwargs):
        return self.connections_to_matrix(self.connections(network, p=p), **kwargs)

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
    def create(cls, network, n_observations, weighter_class):
        return cls(
            network.extent,
            network.get_masts(),
            random_locations(network.extent, n_observations),
            weighter_class=weighter_class,
        )

    def plot(self, ax, network=None):
        if network is None:
            colors = 'b.'
            cellcolors = itertools.repeat('#bbbbbb')
        else:
            connections = self.connections(network)
            colors = ['C' + str(i % 10) for i in connections]
            cellcolors = colors
        self.weighter.plot(ax, cellcolors)
        ax.scatter(
            self.observations[:,0],
            self.observations[:,1],
            c=colors,
            s=5 * self.weights
        )
        if network is not None:
            network.plot(ax)

    def plot3d(self, ax, network):
        strengths = self.strengths(network)
        for i, antstrengths in enumerate(strengths):
            ax.scatter(
                self.observations[:,0],
                self.observations[:,1],
                numpy.log(antstrengths),
                c=('C' + str(i)),
                s=9
            )


class ObservationWeighter:
    def __init__(self, observations, extent):
        self.extent = extent
        self.observations = observations
        self.weights = self._calculate_weights()

    def plot(self, ax, colors):
        pass


class VoronoiAreaObservationWeighter(ObservationWeighter):
    def _calculate_weights(self):
        self.cells = list(voronoi.cells(self.observations, self.extent))
        return numpy.array([cell.area for cell in self.cells])

    def plot(self, ax, colors):
        for cell, color in zip(self.cells, colors):
            ax.plot(*cell.exterior.xy, color=color, lw=0.25)


class ClusteredRandomObservationWeighter(ObservationWeighter):
    def __init__(self, *args, weight_var=0.25, clustering=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_var = weight_var
        self._weight_var = scipy.stats.norm(scale=self.weight_var)
        self.clustering = clustering
        self.range = self._dispersion() / self.clustering
        self._dist_var = scipy.stats.norm(scale=self.range)

    def _calculate_weights(self):
        n_observations = self.observations.shape[0]
        weights = numpy.ones(n_observations) / n_observations
        for i in range(n_observations):
            centroid = (
                self.observations
                * (weights + _weight_var.rvs(n_observations)).reshape(-1, 1)
            ).sum(axis=1)
            dists = numpy.linalg.norm(self.observations - centroid, axis=0)
            weights += self._dist_var.pdf(dists)
            weights /= weights.sum()
        return weights

    def _dispersion(self):
        centroid = self.observations.mean(axis=1)
        return numpy.linalg.norm(self.observations - centroid, axis=0).mean()



class AntennaNetworkEstimator:
    def estimate(self, system, connections):
        raise NotImplementedError

    def _network_from_params(self, system, params, antenna_class=None):
        if antenna_class is None:
            antenna_class = system.get_antenna_class()
        return AntennaNetwork(system.extent, [antenna_class(
                mast=mast,
                **{
                    key : values[i]
                    for key, values in params.items()
                }
            )
            for i, mast in enumerate(system.masts)
        ])


class MeasureNetworkEstimator(AntennaNetworkEstimator):
    # UNIT_KAPPA = 

    def estimate(self, system, connections=None, connmatrix=None, network=None):
        # strengths assumed equal
        if connmatrix is None:
            if connections is None:
                if network is None:
                    raise ValueError('must provide connection indices or matrix')
                else:
                    connmatrix = system.connection_matrix(network, p=1)
            else:
                connmatrix = system.connections_to_matrix(connections, weighted=True)
        params = self.calculate_parameters(system, connmatrix)
        params['strength'] = numpy.ones(system.n_masts)
        return self._network_from_params(system, params)

    @classmethod
    def calculate_parameters(cls, system, connmatrix, names=None):
        if names is None:
            names = system.get_param_names()
        connected_weight = connmatrix.sum(axis=1)
        assert (connected_weight > 0).all()
        print(connmatrix)
        print(connmatrix.shape)
        print('cw')
        print(connected_weight)
        # print(system.unitvectors)
        print(system.unitvectors.shape)
        if 'narrowness' in names or 'principal_angle' in names:
            sincossums = (
                (system.unitvectors * connmatrix[:,:,numpy.newaxis]).sum(axis=1)
                / connected_weight[:,numpy.newaxis]
            )
            print(system.unitvectors)
            print('sc')
            print(connected_weight[:,numpy.newaxis])
            print(sincossums)
            sincossums = numpy.where(numpy.isnan(sincossums), 0, sincossums)
            print(vector.length(sincossums))
        params = {}
        if 'range' in names:
            params['range'] = (
                (system.distances * connmatrix).sum(axis=1)
                / connected_weight
            )
        if 'narrowness' in names:
            sumlengths = vector.length(sincossums)
            sumlengths_one = sumlengths == 1
            params['narrowness'] = cls._estimate_kappa(
                numpy.where(sumlengths_one, 0, sumlengths)
            )
        if 'principal_angle' in names:
            params['principal_angle'] = vector.angle(sincossums)
        return params

    @staticmethod
    def _estimate_kappa(rbar, n_iter=3):
        print(rbar)
        rbarsq = rbar ** 2
        kappa = rbar * (2 - rbarsq) / (1 - rbarsq)
        print(kappa)
        for i in range(n_iter):
            apkappa = scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)
            apkappadelta = numpy.where(kappa == 0, 0, apkappa)
            kappa -= (apkappa - rbar) / (1 - apkappa ** 2 - apkappadelta)
        return kappa


class AdjustingNetworkEstimator(AntennaNetworkEstimator):
    def __init__(self, rate=0.25, tol=1e-6, maxiter=1000):
        self.rate = rate
        self.tol = tol
        self.maxiter = maxiter

    def estimate(self, system, connections):
        measurer = MeasureNetworkEstimator()
        true_connmatrix = system.connections_to_matrix(connections, weighted=True)
        true_params = measurer.calculate_parameters(system, true_connmatrix)
        true_totweights = true_connmatrix.sum(axis=1)
        # estimate initial guess using measurer
        prev_params = true_params.copy()
        prev_params['strength'] = numpy.ones(system.n_masts)
        est_net = self._network_from_params(system, prev_params)
        prev_fit = -1
        n_iter = 0
        while n_iter <= self.maxiter:
            est_connections = est_net.connections(system.observations)
            est_connmatrix = system.connections_to_matrix(est_connections, weighted=True)
            fit = numpy.sqrt(true_connmatrix * est_connmatrix).sum() / true_connmatrix.sum()
            print(fit)
            if (fit - prev_fit) < 1e-6:
                break
            curest_params = measurer.calculate_parameters(system, est_connmatrix)
            curest_totweights = est_connmatrix.sum(axis=1)
            est_params = {
                key : prev_params[key] + self.rate * (true_params[key] - curest_params[key])
                for key in curest_params
            }
            est_params['strength'] = (
                prev_params['strength']
                + self.rate * (true_totweights - curest_totweights) / curest_totweights
            )
            # strength adjusted to approach the target observation weight sum
            est_net = self._network_from_params(system, est_params)
            n_iter += 1
            prev_fit = fit
        return est_net


class AntennaNetworkRecombiner:
    def __init__(self, sigma=0.1):
        self.sigma = 0.1
        self._breakpoint = scipy.stats.truncnorm(a=0, b=1, loc=0.5, scale=self.sigma)

    def recombine(self, net1, net2):
        n_antennas = len(net1.antennas)
        n1 = int(self._breakpoint.rvs(1) * n_antennas)
        antennas1 = random.sample(net1.antennas, n1)
        masts1 = set(antenna.mast for antenna in antennas1)
        antennas2 = [
            antenna for antenna in net2.antennas
            if antenna.mast not in masts1
        ]
        return AntennaNetwork(net1.extent, antennas1 + antennas2)


# class AntennaNetworkMutator:

