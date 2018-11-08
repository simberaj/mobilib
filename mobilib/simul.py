
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
    def __init__(self, location, angle_deg):
        super().__init__(location)
        self.angle_deg = angle_deg
        self.angle = numpy.radians(self.angle_deg)

    def get_angle(self):
        return self.angle


class Antenna:
    PARAM_FORMAT = '{0.strength:.2f}/{0.range:.2f}@{0.principal_angle_deg:.0f}w{0.narrowness:.2f}'
    ALL_FORMAT = (
        '<Antenna({0.mast.location[0]:.2f},{0.mast.location[1]:.2f};'
        + PARAM_FORMAT + ')'
    )
    VAR_NAMES = NotImplemented

    def __init__(self, mast, strength, range, narrowness, strength_sigma=0):
        self.mast = mast
        self.strength = strength
        self.strength_sigma = strength_sigma
        self.range = range
        self.narrowness = narrowness
        self._update_distros()

    @property
    def location(self):
        return self.mast.location

    @property
    def principal_angle_deg(self):
        return float(numpy.degrees(self.principal_angle))

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

    def plot_annotation(self, ax):
        label = self.PARAM_FORMAT.format(self)
        ax.annotate(label, xy=self.location, xytext=self.location)
        
    def get_parameters(self):
        return tuple(self.__dict__[param] for param in self.VAR_NAMES)

    def __repr__(self):
        return self.ALL_FORMAT.format(self)


class VariableAngleAntenna(Antenna):
    VAR_NAMES = ['strength', 'range', 'narrowness', 'principal_angle']

    def __init__(self, *args, **kwargs, principal_angle=0):
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
    if isinstance(extent, shapely.geometry.base.BaseGeometry):
        extent = extent
    else:
        extent = shapely.geometry.box(*extent)
    extent_prep = shapely.prepared.prep(extent)
    locs = []
    for loc in generator:
        if extent_prep.contains(shapely.geometry.Point(*location)):
            locs.append(location)
            if len(locs) == n:
                break
    return numpy.array(locs)

    
def random_locations_infinite(extent):
    xmin, ymin, xmax, ymax = extent.bounds
    xlocgen = scipy.stats.uniform(loc=xmin, scale=xmax-xmin)
    ylocgen = scipy.stats.uniform(loc=ymin, scale=ymax-ymin)
    while True:
        yield numpy.hstack([xlocgen.rvs(1), ylocgen.rvs(1)])

        
def rectgrid_locations_all(extent, grid_size):
    xmin, ymin, xmax, ymax = extent.bounds
    xeq = (xmax - xmin - ((xmax - xmin) // grid_size) * grid_size) / 2
    yeq = (ymax - ymin - ((ymax - ymin) // grid_size) * grid_size) / 2
    xs = numpy.arange(xmin + xeq, xmax, grid_size)
    ys = numpy.arange(ymin + yeq, ymax, grid_size)
    for x in xs:
        for y in ys:
            yield numpy.array([x,y])



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
        for location in random_locations(n_antennas):
            yield VariableAngleAntenna(
                mast=Mast(location),
                strength_sigma=self.strength_sigma,
                strength=self._strength_gen.rvs(1),
                range=self._range_gen.rvs(1),
                narrowness=self._narrowness_gen.rvs(1),
                principal_angle=self._angle_gen.rvs(1),
            )


class FixedAngleNetworkGenerator(NetworkGenerator):
    def __init__(self, *args, **kwargs, grouping_factor):
        super().__init__(*args, **kwargs)
        self.grouping_factor = grouping_factor

    def _generate_antennas(self, extent, n_antennas):
        n_masts = int(numpy.ceil(n_antennas / grouping_factor))
        antenna_masts = (numpy.random.rand(n_antennas) * n_masts).astype(int)
        mast_antenna_counts = [(antenna_masts == i).sum() for i in range(n_masts)]
        mast_locs = random_locations(extent, n_masts)
        for mast_i in antenna_masts:
            if mast_antenna_counts[mast_i] == 1:
                narrowness = 0
            else:
                narrowness = self._narrowness_gen.rvs(1)
            yield FixedAngleAntenna(
                mast=DirectedMast(mast_locs[mast_i], self._angle_gen.rvs(1)),
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

    def connections(self, locations):
        return self.strengths_to_connections(self.strengths(locations))

    def connections_from_distangles(self, distances, angles):
        return self.strengths_to_connections(
            self.strengths_from_distangles(distances, angles)
        )

    def get_parameters(self):
        return zip(*[antenna.get_parameters() for antenna in self.antennas])

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
    # TODO here!
    def __init__(self, extent, masts, observations):
        self.extent = extent
        self.masts = masts
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

    def strengths(self, network):
        return network.strengths_from_distangles(self.distances, self.angles)

    def connections(self, network):
        return network.connections_from_distangles(self.distances, self.angles)

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
            connections = self.connections(network)
            colors = ['C' + str(i) for i in connections]
            cellcolors = colors
        for cell, color in zip(self.cells, cellcolors):
            ax.plot(*cell.exterior.xy, color=color, lw=0.25)
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


class AntennaNetworkEstimator:
    PARAM_NAMES = ['strength', 'range', 'principal_angle', 'narrowness']

    def estimate(self, system, connections):
        raise NotImplementedError

    def _network_from_params(self, system, *args, **kwargs):
        kwargs.update({
            name : value
            for name, value in zip(self.PARAM_NAMES, args)
        })
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
        # strengths assumed equal
        strengths = numpy.ones(system.n_masts)
        connmatrix = system.connections_to_matrix(connections, weighted=True)
        params = (strengths, ) + self.calculate_parameters(system, connmatrix)
        return self._network_from_params(system, *params)

    @classmethod
    def calculate_parameters(cls, system, connmatrix):
        sincossums = (
            (system.unitvectors * connmatrix[:,:,numpy.newaxis]).sum(axis=1)
            / connmatrix.sum(axis=1)[:,numpy.newaxis]
        )
        # range is mean distance of connected point from antenna
        ranges = (system.distances * connmatrix).sum(axis=1) / connmatrix.sum(axis=1)
        # principal angle is mean angle of connected point from antenna
        princ_angles = vector.angle(sincossums)
        narrownesses = cls._estimate_kappa(vector.length(sincossums))
        return ranges, princ_angles, narrownesses

    @staticmethod
    def _estimate_kappa(rbar, n_iter=3):
        rbarsq = rbar ** 2
        kappa = rbar * (2 - rbarsq) / (1 - rbarsq)
        for i in range(n_iter):
            apkappa = scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)
            kappa -= (apkappa - rbar) / (1 - apkappa ** 2 - apkappa / kappa)
        return kappa


class AdjustingNetworkEstimator(AntennaNetworkEstimator):
    def __init__(self, rate=0.25, tol=1e-6, maxiter=1000):
        self.rate = rate
        self.tol = tol
        self.maxiter = maxiter

    def estimate(self, system, connections):
        measurer = MeasureNetworkEstimator()
        true_connmatrix = system.connections_to_matrix(connections, weighted=True)
        true_ranges, true_angles, true_kappas = measurer.calculate_parameters(system, true_connmatrix)
        true_totweights = true_connmatrix.sum(axis=1)
        # estimate initial guess using measurer
        est_net = measurer.estimate(system, connections)
        est_connections = est_net.connections(system.observations)
        est_connmatrix = system.connections_to_matrix(est_connections, weighted=True)
        prev_fit = -1
        fit = numpy.sqrt(true_connmatrix * est_connmatrix).sum() / true_connmatrix.sum()
        n_iter = 0
        while (fit - prev_fit) > self.tol and n_iter <= self.maxiter:
            print(fit)
            est_ranges, est_angles, est_kappas = measurer.calculate_parameters(system, est_connmatrix)
            # strength adjusted to approach the target observation weight sum
            est_totweights = est_connmatrix.sum(axis=1)
            diffs = self.rate * numpy.stack([
                (true_totweights - est_totweights) / est_totweights,
                true_ranges - est_ranges,
                true_angles - est_angles,
                true_kappas - est_kappas,
            ])
            est_net = self._network_from_params(system, *(
                numpy.array(curparams) + paramdiffs
                for curparams, paramdiffs in zip(
                    est_net.get_parameters(),
                    diffs
                )
            ))
            est_connections = est_net.connections(system.observations)
            est_connmatrix = system.connections_to_matrix(est_connections, weighted=True)
            prev_fit = fit
            fit = numpy.sqrt(true_connmatrix * est_connmatrix).sum() / true_connmatrix.sum()
            n_iter += 1
        print(fit)
        return est_net