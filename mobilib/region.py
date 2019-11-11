
import logging

import numpy as np
import pandas as pd


class EvaluatorInterface:
    def feed(self, interactions, unit_props):
        raise NotImplementedError

    # def eval_one(self, region_units, core_units):
        # raise NotImplementedError

    def eval_all(self, regions, cores):
        raise NotImplementedError



class UnitCountEvaluator:
    def feed(self, interactions, unit_props):
        pass

    # def eval_one(self, region_units, core_units):
        # return len(region_units)

    def eval_all(self, regions, cores):
        return regions.value_counts()


class SourceFlowSumEvaluator:
    def feed(self, interactions, unit_props):
        self.flowsums = interactions.groupby(level=0).sum()

    def eval_all(self, regions, cores):
        return self.flowsums.groupby(regions).sum()


class PropertySumEvaluator:
    def __init__(self, criterion):
        self.criterion = criterion

    def feed(self, interactions, unit_props):
        try:
            self.prop = unit_props[self.criterion]
        except KeyError as err:
            raise LookupError(f'{self.criterion} unit property not specified') from err

    # def eval_one(self, region_units, core_units):
        # return self.props[region_units].sum()

    def eval_all(self, regions, cores):
        return self.prop.groupby(regions).sum()


PARAMETERLESS_EVALUATORS = {
    'unit_count': UnitCountEvaluator(),
    'sourceflow_sum': SourceFlowSumEvaluator(),
}


def evaluator(criterion):
    if criterion in PARAMETERLESS_EVALUATORS:
        return PARAMETERLESS_EVALUATORS[criterion]
    else:
        return PropertySumEvaluator(criterion)


class VerifierInterface:
    def verify(self, value):
        raise NotImplementedError


class YesmanVerifier:
    def verify(self, value):
        return True


class MinimumVerifier:
    def __init__(self, threshold):
        self.threshold = threshold

    def verify(self, value):
        return value >= self.threshold


class TargeterInterface:
    pass


class InteractionTargeter:
    def __init__(self, source_core=True, target_core=True):
        self.source_core = source_core
        self.target_core = target_core

    def feed(self, interactions, unit_props):
        self.interactions = interactions

    def target(self, units, regions, cores):
        strengths = self._get_strengths(units, regions, cores)
        if strengths.empty:
            return np.nan
        else:
            return strengths.groupby(level=1).sum().idxmax()

    def targets(self, units, regions, cores):
        strengths = self._get_strengths(units, regions, cores)
        if strengths.empty:
            return pd.Series(np.nan, index=units)
        else:
            return strengths.groupby(level=0).idxmax()

    def _get_strengths(self, units, regions, cores):
        if self.source_core:
            # select only core units for interaction sources
            sources = cores[units][cores].index
        else:
            sources = units
        # interactions to targets outside the given units
        strengths = self.interactions.loc[sources].drop(units, level=1)
        if strengths.empty:
            return strengths
        target_units = strengths.index.get_level_values(1)
        if self.target_core:
            strengths *= cores[target_units]
        # group targets by region and return
        regs = strengths.reset_index(level=1)
        target_col, value_col = regs.columns
        regs[target_col] = regs[target_col].map(regions[target_units])
        return regs.set_index(target_col, append=True)[value_col].groupby(level=(0,1)).sum()


class AggregatorInterface:
    def aggregate(self, regions, cores):
        raise NotImplementedError


class StepwiseAggregator:
    def __init__(self, evaluator, verifier, targeter, dissolve_region=False):
        self.evaluator = evaluator
        self.verifier = verifier
        self.targeter = targeter
        self.dissolve_region = dissolve_region

    def aggregate(self, regions, cores):
        logging.debug('launching aggregation')
        regions = regions.copy()
        cores = cores.copy()
        value_ser = self.evaluator.eval_all(regions, cores)
        logging.debug('initial evaluation of %d regions', len(value_ser.index))
        for id, value in value_ser.items():
            logging.debug('  %s: %g', id, value)
        while True:
            aggreg_code = value_ser.idxmin()
            if self.verifier.verify(value_ser[aggreg_code]):
                logging.debug('region %s stable at %g, terminating', aggreg_code, value_ser[aggreg_code])
                break
            else:   # aggregate the bastard
                logging.debug('aggregating region %s with value %g', aggreg_code, value_ser[aggreg_code])
                aggreg_units = regions[regions == aggreg_code].index
                logging.debug('involved units: %s', ', '.join(str(x) for x in aggreg_units))
                targets = self._get_targets(aggreg_units, regions, cores)
                if targets.isna().any():
                    logging.warning('region %s not aggregable, keeping', aggreg_code)
                    targets.dropna(inplace=True)
                # apply the aggregation
                regions.update(targets)
                cores[targets.index] = False
                # update region values
                selector = regions.isin(targets.unique())
                reevals = self.evaluator.eval_all(
                    regions[selector],
                    cores[selector],
                )
                logging.debug('reevaluating regions')
                for id, value in reevals.items():
                    logging.debug('  %s: %g', id, value)
                value_ser.update(reevals)
                value_ser.drop(aggreg_code, inplace=True)
        return regions, cores
    
    def _get_targets(self, aggreg_units, regions, cores):
        if self.dissolve_region:
            # different target for each unit
            return self.targeter.targets(aggreg_units, regions, cores)
        else:
            # select a single target for all units
            target = self.targeter.target(aggreg_units, regions, cores)
            if not np.isnan(target):
                logging.debug('single target: %s', target)
            return pd.Series(target, index=aggreg_units)
