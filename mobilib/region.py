import sys
import logging
from typing import Any, List, Tuple

import numpy as np
import pandas as pd


class EvaluatorInterface:
    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        raise NotImplementedError

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def get_required_properties(self) -> List[str]:
        raise NotImplementedError

    def get_criteria(self) -> List[str]:
        raise NotImplementedError


class PropertylessEvaluator:
    def feed(self, interactions, unit_props):
        pass

    def eval_all(self, regions, cores):
        return pd.DataFrame(self._compute(regions, cores).rename(self.name))

    def get_required_properties(self):
        return []

    def get_criteria(self):
        return [self.name]


class ConstantEvaluator(PropertylessEvaluator):
    name = 'constant'

    def _compute(self, regions, cores):
        return pd.Series(1, index=regions.unique())


class UnitCountEvaluator(PropertylessEvaluator):
    name = 'unit_count'

    def _compute(self, regions, cores):
        return regions.value_counts(sort=False)


class SourceFlowSumEvaluator(PropertylessEvaluator):
    name = 'sourceflow_sum'

    def feed(self, interactions, unit_props):
        self.flowsums = interactions.groupby(level=0).sum()

    def _compute(self, regions, cores):
        return self.flowsums.groupby(regions).sum()


class PropertySumEvaluator:
    def __init__(self, criterion):
        self.criterion = criterion

    def feed(self, interactions, unit_props):
        try:
            self.prop = unit_props[self.criterion]
        except KeyError as err:
            raise LookupError(f'{self.criterion} unit property not specified') from err

    def eval_all(self, regions, cores):
        return pd.DataFrame(self.prop.groupby(regions).sum().rename(self.criterion))

    def get_required_properties(self):
        return [self.criterion]

    def get_criteria(self):
        return [self.criterion]


class HinterlandSumEvaluator(PropertySumEvaluator):
    PREFIX = 'hinterland_'

    def eval_all(self, regions, cores):
        # print()
        # print('PROP')
        # print(self.prop)
        # print('REGIONS')
        # print(regions)
        # print('CORES')
        # print(cores)
        # print('PROP-REG')
        # print(self.prop.groupby(regions).sum())
        # print('PROP-NONCORE')
        # print(cores.index)
        # print(self.prop.index)
        # print(self.prop[cores.index])
        # # print(self.prop[~cores])
        # print(self.prop[cores.index][~cores].groupby(regions).sum())
        # print('gzjk')
        # print(self.prop.groupby(regions).sum().sub(
            # self.prop[cores.index][cores].groupby(regions).sum(),
            # fill_value=0
        # ).astype(self.prop.dtype).rename(self.criterion))
        # print('asdf')
        return pd.DataFrame(
            self.prop.groupby(regions).sum().sub(
                self.prop[cores.index][cores].groupby(regions).sum(),
                fill_value=0
            ).astype(self.prop.dtype).rename(self.PREFIX + self.criterion)
        )


class CompoundEvaluator:
    def __init__(self, subevals):
        self.subevals = subevals

    def feed(self, interactions, unit_props):
        for subeval in self.subevals:
            subeval.feed(interactions, unit_props)

    def eval_all(self, regions, cores):
        evaluations = self.subevals[0].eval_all(regions, cores)
        for subeval in self.subevals[1:]:
            subevaluation = subeval.eval_all(regions, cores)
            for col in subevaluation.columns:
                evaluations[col] = subevaluation[col]
        return evaluations

    def get_required_properties(self):
        return list(set(
            prop for subeval in self.subevals
                for prop in subeval.get_required_properties()
        ))

    def get_criteria(self):
        crits = []
        for subeval in self.subevals:
            for crit in subeval.get_criteria():
                if crit not in crits:
                    crits.append(crit)
        return crits


PROPERTYLESS_EVALUATORS = {
    c.name : c for c in PropertylessEvaluator.__subclasses__()
}


def evaluator(criterion=[]):
    if not criterion:
        return mobilib.region.ConstantEvaluator()
    elif len(criterion) > 1:
        return CompoundEvaluator([evaluator([critname]) for critname in criterion])
    else:
        criterion = criterion[0]
        if criterion in PROPERTYLESS_EVALUATORS:
            return PROPERTYLESS_EVALUATORS[criterion]()
        elif criterion.startswith(HinterlandSumEvaluator.PREFIX):
            return HinterlandSumEvaluator(criterion[len(HinterlandSumEvaluator.PREFIX):])
        else:
            return PropertySumEvaluator(criterion)


class VerifierInterface:
    def verify(self, value: Any) -> bool:
        raise NotImplementedError


class YesmanVerifier:
    def verify(self, value):
        return True


class MinimumVerifier:
    def __init__(self, threshold):
        self.threshold = threshold

    def verify(self, value):
        return value >= self.threshold


class CompoundVerifier:
    def __init__(self, partials):
        self.partials = partials

    def verify(self, *args):
        return all(
            partial.verify(value)
            for partial, value in zip(self.partials, args)
        )


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
        # try:
        # print(sources)
        # print(self.interactions)
        # print(units)
        strengths = self.interactions.reindex(
            index=sources, level=0
        ).drop(units, level=1, errors='ignore')
        # except FutureWarning:
            # print(strengths.to_dict())
            # print(sources)
            # raise RuntimeError
        if strengths.empty:
            return strengths
        # print(strengths.to_dict())
        target_units = strengths.index.get_level_values(1)
        if self.target_core:
            # print(strengths)
            # print(cores[target_units])
            # print(list(target_units))
            # print([item in cores.index for item in target_units])
            strengths *= cores[target_units].values
        # group targets by region and return
        regs = strengths.reset_index(level=1)
        target_col, value_col = regs.columns
        # print(regs.to_dict())
        # print(target_units)
        # print(regions[target_units].to_dict())
        regs[target_col] = regs[target_col].map(regions[target_units.unique()])
        return regs.set_index(target_col, append=True)[value_col].groupby(level=(0,1)).sum()


class AggregatorInterface:
    def aggregate(self, regions: pd.Series, cores: pd.Series) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        raise NotImplementedError


class StepwiseAggregator:
    def __init__(self, evaluator, verifier, targeter, dissolve_region=False, sort_criterion=None):
        self.evaluator = evaluator
        self.verifier = verifier
        self.targeter = targeter
        self.dissolve_region = dissolve_region
        self.sort_criterion = sort_criterion

    def aggregate(self, regions, cores):
        # logging.debug('launching aggregation')
        regions = regions.copy()
        cores = cores.copy()
        evaluations = self.evaluator.eval_all(regions, cores)
        # print(evaluations)
        # raise RuntimeError
        sort_crit = (
            self.sort_criterion if self.sort_criterion is not None
            else evaluations.columns[0]
        )
        stages = []
        # logging.debug('initial evaluation of %d regions', len(evaluations.index))
        # self._show_evaluations(evaluations)
        while True:
            aggreg_code = evaluations[sort_crit].idxmin()
            # sys.stderr.write(str(aggreg_code) + ' ' + str(sort_crit) + '\n')
            aggreg_eval = tuple(evaluations.loc[aggreg_code,:])
            if self.verifier.verify(*aggreg_eval):
                # this one is stable, set evaluation to infinity and continue
                # logging.debug('region %s stable at %s', aggreg_code, aggreg_eval)
                if not np.isfinite(evaluations.loc[aggreg_code,sort_crit]) or len(aggreg_eval) == 1:
                    # logging.debug('all regions stable, terminating')
                    break
                else:
                    evaluations.loc[aggreg_code,sort_crit] = np.inf
            else:   # aggregate the bastard
                # logging.debug('aggregating region %s with value %s', aggreg_code, aggreg_eval)
                stages.append(aggreg_eval)
                aggreg_units = regions[regions == aggreg_code].index
                # logging.debug('involved units: %s', ', '.join(str(x) for x in aggreg_units))
                targets = self._get_targets(aggreg_units, regions, cores)
                if targets.isna().any():
                    logging.warning('region %s not aggregable, keeping', aggreg_code)
                    targets.dropna(inplace=True)
                # apply the aggregation
                regions.update(targets)
                cores[targets.index] = False
                # update region values
                # logging.debug('reevaluating regions')
                selector = regions.isin(targets.unique())
                reevals = self.evaluator.eval_all(
                    regions[selector],
                    cores[selector],
                )
                # self._show_evaluations(reevals)
                evaluations.update(reevals)
                evaluations.drop(aggreg_code, inplace=True)
        stages.extend(evaluations.itertuples(name=None, index=False))
        # for row in evaluations.itertuples(name=None):
            # print('  %s: %s' % (row[0], ', '.join(str(val) for val in row[1:])))
        stage_df = pd.DataFrame.from_records(stages, columns=evaluations.columns)
        return regions, cores, stage_df

    # def _show_evaluations(self, evaluations):
        # for row in evaluations.itertuples(name=None):
            # print('  %s: %s', row[0], ', '.join([str(val) for val in row[1:]]))
            # logging.debug('  %s: %s', row[0], ', '.join([str(val) for val in row[1:]]))

    def _get_targets(self, aggreg_units, regions, cores):
        if self.dissolve_region:
            # different target for each unit
            return self.targeter.targets(aggreg_units, regions, cores)
        else:
            # select a single target for all units
            target = self.targeter.target(aggreg_units, regions, cores)
            # if not np.isnan(target):
                # logging.debug('single target: %s', target)
            return pd.Series(target, index=aggreg_units)
