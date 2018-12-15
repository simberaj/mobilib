
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
        + PARAM_FORMAT + ')>'
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
        self.update()

    def copy(self):
        return type(self)(
            self.mast,
            strength_sigma=self.strength_sigma,
            **self.get_param_dict(),
        )

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

    def get_param(self, name):
        if name in self.VAR_NAMES:
            return self.__dict__[name]
    
    def get_param_dict(self):
        return {name : self.__dict__[name] for name in self.VAR_NAMES}

    def update(self):
        self._strength = scipy.stats.norm(loc=self.strength, scale=self.strength_sigma).rvs
        # self._distance = scipy.stats.norm(scale=self.range)
        self._distance = scipy.stats.cauchy(scale=self.range).pdf
        if self.principal_angle is None or self.narrowness == 0:
            # self._angle = scipy.stats.uniform(loc=-2 * numpy.pi, scale=4 * numpy.pi)
            constprob = 1 / (2 * numpy.pi)
            self._angle = lambda x: constprob
        else:
            self._angle = scipy.stats.vonmises(
                loc=self.principal_angle,
                kappa=self.narrowness
            ).pdf

    def strengths(self, locations):
        dirvec = self.mast.dirvectors(locations)
        return self.strengths_from_distangles(vector.length(dirvec), vector.angle(dirvec))

    def strengths_from_distangles(self, distances, angles):
        # print(distances)
        # print(angles)
        # import matplotlib.pyplot as plt
        # plt.title(self.PARAM_FORMAT.format(self))
        # plt.scatter(angles, self._strength(len(angles)) * self._angle(angles))
        # plt.show()
        return (
            self._strength(len(distances))
            * self._distance(distances)
            * self._angle(angles)
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
    return filter_by_extent(rectgrid_locations_all(extent, grid_size), extent)


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


def besselrat(kappa):
    return scipy.special.iv(1, kappa) / scipy.special.iv(0, kappa)


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
        strengths = self._strength_gen.rvs(n_antennas)
        strengths /= strengths.sum()
        for i, location in enumerate(random_locations(extent, n_antennas)):
            yield VariableAngleAntenna(
                mast=Mast(location),
                strength_sigma=self.strength_sigma,
                strength=strengths[i],
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

    def get_param(self, name):
        return numpy.array([a.get_param(name) for a in self.antennas])

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
    
    def copy(self):
        return AntennaNetwork(self.extent, [a.copy() for a in self.antennas])

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

    def __init__(self, extent, masts, observations, weighter_class, add_antennas=False):
        self.extent = extent
        self.masts = masts
        self.mast_type = type(self.masts[0])
        # print(set(tuple(mast.location) for mast in self.masts))
        if add_antennas:
            self.observations = numpy.concatenate((
                observations,
                numpy.array(list(set(tuple(mast.location) for mast in self.masts)))
            ))
        else:
            self.observations = observations
        self.dirvectors = numpy.stack([
            mast.dirvectors(self.observations)
            for mast in self.masts
        ])
        self.distances = vector.length(self.dirvectors)
        self.angles = vector.angle(self.dirvectors)
        self.unitvectors = self.dirvectors / numpy.where(self.distances == 0, 1, self.distances)[:,:,numpy.newaxis]
        self.weighter = weighter_class(self.observations, self.extent)
        self.weights = self.weighter.weights

    # @property
    # def weights(self):
        # return self.weighter.weights

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
    def n_observations(self):
        return self.observations.shape[0]

    @property
    def n_masts(self):
        return len(self.masts)

    @classmethod
    def create(cls, network, locations, weighter_class, extent=None, **kwargs):
        if extent is None: extent = network.extent
        return cls(
            extent,
            network.get_masts(),
            locations,
            weighter_class=weighter_class,
            **kwargs,
        )

    @classmethod
    def create_random(cls, network, n_observations, weighter_class, extent=None, **kwargs):
        if extent is None: extent = network.extent
        return cls.create(network,
            random_locations(extent, n_observations),
            weighter_class=weighter_class,
            extent=extent,
            **kwargs,
        )

    @classmethod
    def create_grid(cls, network, gridsize, weighter_class, extent=None, **kwargs):
        if extent is None: extent = network.extent
        return cls.create(network,
            rectgrid_locations(extent, gridsize),
            weighter_class=weighter_class,
            extent=extent,
            **kwargs,
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

    def plot3d(self, ax, network, mode='strength'):
        strengths = self._plotting_strengths(network, mode=mode)
        for i, antstrengths in enumerate(strengths):
            ax.scatter(
                self.observations[:,0],
                self.observations[:,1],
                antstrengths,
                c=('C' + str(i % 10)),
                s=9
            )
    
    def _plotting_strengths(self, network, mode='strength'):
        raws = self.strengths(network)
        if mode == 'strength':
            return numpy.log(raws)
        elif mode == 'dominance':
            return raws / raws.sum(axis=0)
        elif mode == 'anglerel':
            arels = raws * self.distances ** 2
            return arels / arels.mean(axis=1)[:,numpy.newaxis]
        else:
            raise ValueError('unknown strength plotting mode')


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

    @staticmethod
    def _dominances_by_users(user_fs, user_probs):
        return (
            user_fs[:,numpy.newaxis,:] * user_probs[numpy.newaxis,:,:]
        ).sum(axis=-1) / user_probs.sum(axis=1)
    
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
    
    def _network_from_list(self, system, antennas):
        return AntennaNetwork(system.extent, antennas)

    @staticmethod
    def _estimate_kappa(rbar, n_iter=3):
        rbarsq = rbar ** 2
        kappa = rbar * (2 - rbarsq) / (1 - rbarsq)
        for i in range(n_iter):
            apkappa = besselrat(kappa)
            apkappadelta = numpy.where(kappa == 0, 0, apkappa)
            kappa -= (apkappa - rbar) / (1 - apkappa ** 2 - apkappadelta)
        return kappa


class InitialNetworkEstimator(AntennaNetworkEstimator):
    # def estimate(self, system, user_fs, user_probs):
    def estimate(self, system, user_fs):
        # dominances = self._dominances_by_users(user_fs, user_probs)
        return self._network_from_params(system,
            # self.calculate_parameters(system, dominances)
            self.calculate_parameters(system, user_fs)
        )

    @classmethod
    def calculate_parameters(cls, system, dominances, names=None):
        if names is None:
            names = system.get_param_names()
        domination_sum = dominances.sum(axis=1)
        if 'narrowness' in names or 'principal_angle' in names:
            sincossums = (
                system.unitvectors * dominances[:,:,numpy.newaxis]
            ).sum(axis=1)
        params = {}
        if 'range' in names:
            mdist = system.distances.mean()
            params['range'] = (
                (system.distances * dominances).sum(axis=1)
                / (domination_sum * mdist)
            )
        if 'strength' in names:
            params['strength'] = (
                (system.weights[numpy.newaxis,:] * dominances).sum(axis=1)
                / system.weights.sum()
            )
            params['strength'] /= params['strength'].sum()
        if 'narrowness' in names:
            rbars = vector.length(sincossums) / domination_sum
            params['narrowness'] = cls._estimate_kappa(rbars)
        if 'principal_angle' in names:
            params['principal_angle'] = vector.angle(sincossums)
        return params

        
class SignalBasedNetworkEstimator(AntennaNetworkEstimator):
    CAP_N = 5
    
    def _estimate_antenna(self, mast, signals, distances, unitvectors, obs, names, antenna_class):
        sqdists = distances ** 2
        distagnosts = signals * sqdists
        params = {}
        if 'principal_angle' in names:
            cap_is = numpy.argpartition(distagnosts, -self.CAP_N)[-self.CAP_N:]
            alphavec = unitvectors[cap_is,:].mean(axis=0)
            cosalpha, sinalpha = alphavec / vector.length(alphavec)
            params['principal_angle'] = (
                numpy.arctan2(sinalpha, cosalpha) % (2 * numpy.pi)
            )
        else:
            alpha = mast.get_angle()
            cosalpha = numpy.cos(alpha)
            sinalpha = numpy.sin(alpha)
        q = cosalpha * unitvectors[:,0] + sinalpha * unitvectors[:,1]
        kappa = self._find_optimal(
            (lambda kappa: numpy.exp(kappa * q) / scipy.special.iv(0, kappa)),
            targets=distagnosts,
            maxval=42,
        )
        besselkappa = scipy.special.iv(0, kappa)
        angexp = numpy.exp(kappa * q)
        params['narrowness'] = kappa
        angagnosts = signals * besselkappa / angexp
        gamma = self._find_optimal(
            (lambda gamma: gamma / (sqdists + gamma ** 2)),
            targets=angagnosts,
            maxval=distances.max()
        )
        unscaled = gamma * angexp / (
            2 * numpy.pi ** 2 * besselkappa * (sqdists + gamma ** 2)
        )
        params['range'] = gamma
        params['strength'] = (signals / unscaled).mean()
        return antenna_class(mast, **params)
    
    @staticmethod
    def _find_optimal(rawfx, targets, maxval):
        tgtmean = targets.mean()
        def errf(param):
            raws = rawfx(param)
            raws *= (tgtmean / raws.mean())
            return ((raws - targets) ** 2).sum()
        return scipy.optimize.minimize_scalar(
            errf, method='bounded', bounds=(0, maxval)
        ).x
        
    def estimate(self, system, signals, names=None):
        if names is None:
            names = system.get_param_names()
        return self._network_from_list(system, [
            self._estimate_antenna(
                mast=system.masts[i],
                signals=signals[i],
                distances=system.distances[i],
                unitvectors=system.unitvectors[i],
                obs=system.observations,
                names=names,
                antenna_class=system.get_antenna_class(),
            )
            for i in range(system.distances.shape[0])
        ])
        
        # angulars = signals * system.distances ** 2
        # n_cap = 5
        # cap_is = numpy.argpartition(angulars, -n_cap, axis=1)[:,-n_cap:]
        # n_antennas = system.distances.shape[0]
        # for i in range(n_antennas):
            
            # cosfis = system.unitvectors[i,:,0]
            # sinfis = system.unitvectors[i,:,1]
            # dists = system.distances[i]
            # sqdists = dists ** 2
            # sigs = signals[i]
            
        
            # alphavec = system.unitvectors[i,cap_is[i],:].mean(axis=0)
            # cosalpha, sinalpha = alphavec / vector.length(alphavec)
            # q = cosalpha * cosfis + sinalpha * sinfis
            # # kappas = numpy.linspace(0, 10, 200)
            # angs = angulars[i]
            # angmean = angulars[i].mean()
            # def errf(kappa):
                # raws = numpy.exp(kappa * q) / scipy.special.iv(0, kappa)
                # raws *= (angmean / raws.mean())
                # return ((raws - angs) ** 2).sum()
            # kappa = scipy.optimize.minimize_scalar(errf, method='bounded', bounds=(0, 42)).x
            # besselkappa = scipy.special.iv(0, kappa)
            # angexp = numpy.exp(kappa * q)
            # distsigs = sigs * besselkappa / angexp
            # distsigmean = distsigs.mean()
            # def errf_gamma(gamma):
                # raws = gamma / (sqdists + gamma ** 2)
                # raws *= (distsigmean / raws.mean())
                # return ((raws - distsigs) ** 2).sum()
            # gamma = scipy.optimize.minimize_scalar(errf_gamma, method='bounded', bounds=(0, dists.max())).x
            # unscaled = gamma * angexp / (2 * numpy.pi ** 2 * besselkappa * (sqdists + gamma ** 2))
            # P = (sigs / unscaled).mean()
            # print(P, gamma, numpy.degrees(vector.angle(alphavec)), kappa)
            # # print(gamma, numpy.degrees(vector.angle(alphavec)), kappa)
            # # # errs = []
            # # # for kappa in kappas:
                # # # errs.append(errf(kappa))
            # # import matplotlib.pyplot as plt
            # # from mpl_toolkits.mplot3d import Axes3D
            # # fig = plt.figure()
            # # ax = fig.add_subplot(111, projection='3d')
            # # ax.plot(system.observations[:,0], system.observations[:,1], numpy.log(sigs))
            # # ax.plot(system.observations[:,0], system.observations[:,1], numpy.log(unscaled * P))
            # # print(sigs / unscaled)
            # # plt.scatter(dists, numpy.log(distsigs))
            # # # plt.scatter(kappas, errs)
            # # # plt.scatter(system.distances[i], numpy.log(signals[i]))
            # # # plt.scatter(system.angles[i], signals[i] * system.distances[i] ** 2)
            # # # plt.scatter(vector.angle(alphavec), signals[i].mean())
            # plt.show()
        # raise RuntimeError
        
        
            # # def errf(x0):
                # # P, gamma, kappa, alpha = x0
                # # return ((P * gamma * numpy.exp(kappa * (cosfis * numpy.cos(alpha) + sinfis * numpy.sin(alpha))) / (2 * numpy.pi * scipy.special.iv(0, kappa) * (sqdists + gamma ** 2)) - sigs) ** 2).sum()
            # # optres = scipy.optimize.minimize(errf,
                # # (1 / n_antennas, 1, 0, 0),
                # # bounds=[
                    # # (0, 1),
                    # # (0, dists.max()),
                    # # (0, 42),
                    # # (0, 2 * numpy.pi)
                # # ],
                # # method='SLSQP'
            # # )
            # # print(optres)
            # # raise RuntimeError
            
        # if 'narrowness' in names or 'principal_angle' in names:
            # sincossums = (
                # system.unitvectors * weights[:,:,numpy.newaxis]
            # ).sum(axis=1)
        # params = {}
        # if 'narrowness' in names:
            # # print(vector.length(sincossums / weight_sums[:,numpy.newaxis]))
            # # print(vector.length(system.unitvectors.sum(axis=1) / system.n_observations))
            # # print((
                # # vector.length(sincossums / weight_sums[:,numpy.newaxis])
                # # / vector.length(system.unitvectors.sum(axis=1) / system.n_observations)
            # # ))
            # rbars = vector.length(sincossums / weight_sums[:,numpy.newaxis])
            # # rbars = (
                # # vector.length(sincossums / weight_sums[:,numpy.newaxis])
                # # / vector.length(system.unitvectors.sum(axis=1) / system.n_observations)
            # # )
            # params['narrowness'] = cls._estimate_kappa(rbars, n_iter=10)
            # # auxkappa = cls._estimate_kappa(vector.length(system.unitvectors.sum(axis=1) / system.n_observations), n_iter=10)
        # if 'principal_angle' in names:
            # params['principal_angle'] = vector.angle(sincossums)
        # print(params['narrowness'])
        # # print(auxkappa)
        # # print(params['narrowness'] / (1 + auxkappa))
        # # print(cls._estimate_kappa(, n_iter=10))
        # print(numpy.degrees(params['principal_angle']))
        # raise RuntimeError
        # if 'range' in names:
            # mdist = system.distances.mean()
            # params['range'] = (
                # (system.distances * dominances).sum(axis=1)
                # / (domination_sum * mdist)
            # )
        # if 'strength' in names:
            # params['strength'] = (
                # (system.weights[numpy.newaxis,:] * dominances).sum(axis=1)
                # / system.weights.sum()
            # )
            # params['strength'] /= params['strength'].sum()
        # return params


class DummyNetworkEstimator(AntennaNetworkEstimator):
    def estimate(self, system):
        return self._network_from_params(system,
            self.calculate_parameters(system)
        )

    @classmethod
    def calculate_parameters(cls, system, names=None):
        if names is None:
            names = system.get_param_names()
        params = {}
        unit = numpy.ones(system.n_masts)
        if 'range' in names:
            params['range'] = unit * system.distances.mean()
        if 'strength' in names:
            params['strength'] = unit / system.n_masts
        if 'narrowness' in names:
            params['narrowness'] = numpy.zeros(system.n_masts)
        if 'principal_angle' in names:
            params['principal_angle'] = numpy.zeros(system.n_masts)
        return params



class EMNetworkEstimator(AntennaNetworkEstimator):
    POSITIVE_PARAMS = ['strength', 'range', 'narrowness']
    ALL_PARAMS = ['strength', 'range', 'narrowness', 'principal_angle']

    def __init__(self, rate=0.05):
        self.rate = rate

    def _adjust_net(self, system, net, prevsig, newsig, names=None):
        strengths = []
        if names is None:
            names = system.get_param_names()
        for i, antenna in enumerate(net.antennas):
            At, b = self._netcoefs(
                system.distances[i],
                system.unitvectors[i],
                newsig[i],
                prevsig[i],
                antenna,
                names=names
            )
            linopt_diffs = dict(zip(
                names,
                numpy.linalg.inv(At.dot(At.T)).dot(At).dot(b)
            ))
            strength = self._update_antenna(antenna, linopt_diffs)
            if strength is not None:
                strengths.append(strength)
        if strengths:
            strengths = numpy.array(strengths)
            strengths /= strengths.sum()
            print(numpy.round(strengths, 5))
            if strengths.max() > .9:
                raise RuntimeError
            for i, antenna in enumerate(net.antennas):
                antenna.strength = strengths[i]
                antenna.update()
    
    def _update_antenna(self, antenna, opt_diffs):
        # print(antenna, opt_diffs)
        updated_names = [name for name in self.ALL_PARAMS if name in opt_diffs]
        diffarr = numpy.array([opt_diffs[name] for name in updated_names])
        curarr = numpy.array([antenna.get_param(name) for name in updated_names])
        must_be_pos = numpy.array([
            name in self.POSITIVE_PARAMS
            for name in updated_names
        ])
        finarr = curarr + self.rate * diffarr
        minposi = numpy.where(must_be_pos, finarr, numpy.inf).argmin()
        if finarr[minposi] <= 0:
            sel_adj_rate = must_be_pos & (diffarr != 0) & (finarr <= 0)
            # diffarr[minposi] != 0 because it was not negative in curarr
            adj_rate = (curarr[sel_adj_rate] / (2 * -diffarr[sel_adj_rate])).min()
            print('AR', adj_rate)
            finarr = curarr + adj_rate * diffarr
        finvals = dict(zip(updated_names, finarr))
        # print(finvals)
        if 'range' in finvals:
            antenna.range = finvals['range']
        if 'narrowness' in finvals:
            antenna.narrowness = finvals['narrowness']
        if 'principal_angle' in finvals:
            antenna.principal_angle = finvals['principal_angle'] % (2 * numpy.pi)
        # print(antenna)
        if 'strength' in finvals:
            if finvals['strength'] < 0:
                print(updated_names)
                print(diffarr)
                print(curarr)
                print(must_be_pos)
                print(minposi)
                print(finvals)
                raise RuntimeError
            # return finvals['strength'] / finvals['range']
            return finvals['strength']
        else:
            return None

    @staticmethod
    def _netcoefs(distances, unitvectors, newsig, prevsig, antenna, names):
        coefs = []
        if 'narrowness' in names or 'principal_angle' in names:
            sinalpha = numpy.sin(antenna.principal_angle)
            cosalpha = numpy.cos(antenna.principal_angle)
            sinfis = unitvectors[:,1]
            cosfis = unitvectors[:,0]
        if 'strength' in names:
            coefs.append(
                numpy.ones_like(distances)
                # / (antenna.strength * antenna.range)
                / antenna.strength
            )
        if 'range' in names:
            sqdist = distances ** 2
            sqgamma = antenna.range ** 2
            coefs.append(
                (sqdist - sqgamma)
                # -2 * antenna.range
                / (antenna.range * (sqdist + sqgamma))
            )
        if 'narrowness' in names:
            coefs.append(
                cosfis * cosalpha + sinfis * sinalpha
                - besselrat(antenna.narrowness)
            )
        if 'principal_angle' in names:
            coefs.append(
                antenna.narrowness
                * (sinfis * cosalpha - cosfis * sinalpha)
            )
        return numpy.stack(coefs), newsig / prevsig - 1


class EMFixdomNetworkEstimator(EMNetworkEstimator):
    def __init__(self, maxiter=100, maxdrops=10, **kwargs):
        super().__init__(**kwargs)
        self.maxiter = maxiter
        self.maxdrops = maxdrops

    @staticmethod
    def _calibrate_signals(prevs, dominances):
        return dominances * prevs.sum(axis=0)[numpy.newaxis,:]
        
    @staticmethod
    def _dominances_by_signal(signals):
        return signals / signals.sum(axis=0)
        
    @staticmethod
    def dominance_agreement(realdoms, modeldoms):
        return 1 - numpy.sqrt(((realdoms - modeldoms) ** 2).mean())

    def estimate(self, system, user_fs, user_probs):
        # user_fs.shape: n_a, n_u
        # user_probs.shape: n_p, n_u
        net = InitialNetworkEstimator().estimate(system, user_fs, user_probs)
        agr = 0
        best_agr = 0
        n_drops = 0
        dominances = self._dominances_by_users(user_fs, user_probs)
        for i in range(self.maxiter):
            # print(i, net)
            signals = system.strengths(net)
            prev_agr = agr
            agr = self.dominance_agreement(
                dominances,
                self._dominances_by_signal(signals)
            )
            if prev_agr > agr:
                n_drops += 1
                if n_drops >= self.maxdrops:
                    net = best_past_net
                    break
                elif net is best_past_net:
                    net = net.copy()
            else:
                if agr > best_agr:
                    best_past_net = net
                    best_agr = agr
                n_drops = 0
            print(i, agr)
            calib_signals = self._calibrate_signals(signals, dominances)
            self._adjust_net(system, net, signals, calib_signals)
            print(net)
        print()
        print(best_agr)
        print(net)
        return net


class EMVardomNetworkEstimator(EMNetworkEstimator):
    def __init__(self, crowding_tolerance=1, **kwargs):
        super().__init__(**kwargs)
        self.crowding_tolerance = crowding_tolerance

    def estimate(self, place_system, user_fs):
        dummy_net = DummyNetworkEstimator().estimate(place_system)
        user_probs = self._compute_user_probs(place_system.strengths(dummy_net), user_fs)
        print('UP', user_probs)
        dominances = self._compute_dominances(user_fs, user_probs)
        print('domins', dominances)
        net = InitialNetworkEstimator().estimate(place_system, dominances)
        place_weights = place_system.weights * user_probs.sum() / place_system.weights.sum()
        print(net)
        for i in range(4):
            prev_signals = place_system.strengths(net)
            signals = self.calculate_signals(prev_signals, dominances)
            self.adjust_net(place_system, net, prev_signals, signals)
            print(net)
            crowding = self._compute_crowding(user_probs, place_weights)
            # print(crowding)
            user_probs = self._compute_user_probs(
                signals, user_fs, crowding
            )
            dominances = self._compute_dominances(user_fs, user_probs)
        return net

    def _compute_crowding(self, user_probs, place_weights):
        place_probs = user_probs.sum(axis=0)
        # print('PP', place_probs)
        # print('PW', place_weights)
        return (
            1 + (place_weights - place_probs) / (place_weights + place_probs)
        ) ** (1 / self.crowding_tolerance)

    @staticmethod
    def _compute_dominances(user_fs, user_probs):
        return (
            user_fs[:,:,numpy.newaxis] * user_probs[numpy.newaxis,:,:]
        ).sum(axis=1) / user_probs.sum(axis=0)

    @staticmethod
    def _compute_user_probs(place_signals, user_fs, crowding=None):
        print('PS', place_signals)
        # normsigs = place_signals / place_signals.sum(axis=0)[numpy.newaxis,:]
        normsigs = (place_signals ** 2) / (place_signals ** 2).sum(axis=0)[numpy.newaxis,:]
        print('NS', normsigs)
        print(normsigs.shape)
        affinities = (user_fs / normsigs.sum(axis=1)[:,numpy.newaxis]).T.dot(normsigs)
        print('AF', affinities)
        print(affinities.shape)
        # affinities = numpy.sqrt((
            # (user_fs[:,:,numpy.newaxis] - normsigs[:,numpy.newaxis,:]) ** 2
        # ).mean(axis=0))
        if crowding is not None:
            affinities *= crowding[numpy.newaxis,:]
        return affinities / affinities.sum(axis=1)[:,numpy.newaxis]




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
        if 'narrowness' in names or 'principal_angle' in names:
            sincossums = (
                (system.unitvectors * connmatrix[:,:,numpy.newaxis]).sum(axis=1)
                / connected_weight[:,numpy.newaxis]
            )
            sincossums = numpy.where(numpy.isnan(sincossums), 0, sincossums)
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

