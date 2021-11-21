"""Perform classical functional region delimitation."""

import logging
from typing import Any, List, Tuple, Optional, TypeVar

import numpy as np
import pandas as pd


class EvaluatorInterface:
    """Objects that evaluate fitness of regions."""

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        raise NotImplementedError

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        raise NotImplementedError

    def get_required_properties(self) -> List[str]:
        raise NotImplementedError

    def get_criteria(self) -> List[str]:
        raise NotImplementedError


class PropertylessEvaluator(EvaluatorInterface):
    name: str = NotImplemented

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        pass

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(self._compute(regions, cores).rename(self.name))

    def get_required_properties(self) -> List[str]:
        return []

    def get_criteria(self) -> List[str]:
        return [self.name]

    def _compute(self, regions: pd.Series, cores: pd.Series) -> pd.Series:
        raise NotImplementedError


class ConstantEvaluator(PropertylessEvaluator):
    """Evaluate all regions as 1."""
    name = 'constant'

    def _compute(self, regions: pd.Series, cores: pd.Series) -> pd.Series:
        return pd.Series(1, index=regions.unique())


class UnitCountEvaluator(PropertylessEvaluator):
    """Evaluate regions by count of units."""
    name = 'unit_count'

    def _compute(self, regions: pd.Series, cores: pd.Series) -> pd.Series:
        return regions.value_counts(sort=False)


class SourceFlowSumEvaluator(PropertylessEvaluator):
    """Evaluate regions by summing the magnitude of outgoing interactions."""
    name = 'sourceflow_sum'
    flowsums: pd.Series

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        self.flowsums = interactions.groupby(level=0).sum()

    def _compute(self, regions: pd.Series, cores: pd.Series) -> pd.Series:
        return self.flowsums.groupby(regions).sum()


class PropertySumEvaluator(EvaluatorInterface):
    """Evaluate regions by summing a property of their constituent units."""
    prop: pd.Series

    def __init__(self, criterion):
        self.criterion = criterion

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        try:
            self.prop = unit_props[self.criterion]
        except KeyError as err:
            raise LookupError(f'{self.criterion} unit property not specified') from err

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(self.prop.groupby(regions).sum().rename(self.criterion))

    def get_required_properties(self) -> List[str]:
        return [self.criterion]

    def get_criteria(self) -> List[str]:
        return [self.criterion]


class HinterlandSumEvaluator(PropertySumEvaluator):
    """Evaluate regions by summing a property of their non-core units."""
    PREFIX = 'hinterland_'

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            self.prop.groupby(regions).sum().sub(
                self.prop[cores.index][cores].groupby(regions).sum(),
                fill_value=0
            ).astype(self.prop.dtype).rename(self.PREFIX + self.criterion)
        )


class CompoundEvaluator(EvaluatorInterface):
    """Evaluate regions by multiple values or criteria."""

    def __init__(self, subevals: List[EvaluatorInterface]):
        self.subevals = subevals

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        for subeval in self.subevals:
            subeval.feed(interactions, unit_props)

    def eval_all(self, regions: pd.Series, cores: pd.Series) -> pd.DataFrame:
        evaluations = self.subevals[0].eval_all(regions, cores)
        for subeval in self.subevals[1:]:
            subevaluation = subeval.eval_all(regions, cores)
            for col in subevaluation.columns:
                evaluations[col] = subevaluation[col]
        return evaluations

    def get_required_properties(self) -> List[str]:
        return list(set(
            prop
            for subeval in self.subevals
            for prop in subeval.get_required_properties()
        ))

    def get_criteria(self) -> List[str]:
        crits = []
        for subeval in self.subevals:
            for crit in subeval.get_criteria():
                if crit not in crits:
                    crits.append(crit)
        return crits


PROPERTYLESS_EVALUATORS = {
    c.name: c for c in PropertylessEvaluator.__subclasses__()
}


def create_evaluator(criterion: List[str] = []) -> EvaluatorInterface:
    if not criterion:
        return ConstantEvaluator()
    elif len(criterion) > 1:
        return CompoundEvaluator([create_evaluator([critname]) for critname in criterion])
    else:
        criterion = criterion[0]
        if criterion in PROPERTYLESS_EVALUATORS:
            return PROPERTYLESS_EVALUATORS[criterion]()
        elif criterion.startswith(HinterlandSumEvaluator.PREFIX):
            return HinterlandSumEvaluator(criterion[len(HinterlandSumEvaluator.PREFIX):])
        else:
            return PropertySumEvaluator(criterion)


class VerifierInterface:
    """Objects that verify that the evaluation of a region is good enough."""
    def verify(self, value: Any) -> bool:
        raise NotImplementedError


class YesmanVerifier(VerifierInterface):
    """Always make all regions pass the criterion."""

    @staticmethod
    def verify(value: Any) -> bool:
        return True


class MinimumVerifier(VerifierInterface):
    """Only allow those regions with an evaluation at least the given value."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def verify(self, value: float) -> bool:
        return value >= self.threshold


class CompoundAndVerifier(VerifierInterface):
    """Allow only those regions that satisfy all partial verifications."""
    def __init__(self, partials: List[VerifierInterface]):
        self.partials = partials

    def verify(self, *args) -> bool:
        return all(
            partial.verify(value)
            for partial, value in zip(self.partials, args)
        )


class TargeterInterface:
    pass


ID = TypeVar('ID')


class InteractionTargeter:
    """Select aggregation targets for units based on largest interaction."""

    interactions: pd.Series

    def __init__(self, source_core: bool = True, target_core: bool = True):
        self.source_core = source_core
        self.target_core = target_core

    def feed(self, interactions: pd.Series, unit_props: pd.DataFrame) -> None:
        self.interactions = interactions

    def target(self, units: ID, regions: pd.Series, cores: pd.Series) -> ID:
        """Select a target for a single unit."""
        strengths = self._get_strengths(units, regions, cores)
        if strengths.empty:
            return np.nan
        else:
            return strengths.groupby(level=1).sum().idxmax()

    def targets(self, units: pd.Index, regions: pd.Series, cores: pd.Series) -> pd.Series:
        """Select targets for multiple units."""
        strengths = self._get_strengths(units, regions, cores)
        if strengths.empty:
            return pd.Series(np.nan, index=units)
        else:
            return strengths.groupby(level=0).idxmax()

    def _get_strengths(self, units: pd.Index, regions: pd.Series, cores: pd.Series) -> pd.Series:
        if self.source_core:
            # select only core units for interaction sources
            sources = cores[units][cores].index
        else:
            sources = units
        strengths = self.interactions.reindex(
            index=sources, level=0
        ).drop(units, level=1, errors='ignore')
        if strengths.empty:
            return strengths
        target_units = strengths.index.get_level_values(1)
        if self.target_core:
            strengths *= cores[target_units].values
        # group targets by region and return
        regs = strengths.reset_index(level=1)
        target_col, value_col = regs.columns
        regs[target_col] = regs[target_col].map(regions[target_units.unique()])
        return regs.set_index(target_col, append=True)[value_col].groupby(level=(0,1)).sum()


class AggregatorInterface:
    """Objects that aggregate units into regions."""
    def aggregate(self,
                  regions: pd.Series,
                  cores: pd.Series,
                  ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        raise NotImplementedError


class StepwiseAggregator(AggregatorInterface):
    """Aggregate given regions successively until they all satisfy a criterion."""
    def __init__(self,
                 evaluator: EvaluatorInterface,
                 verifier: VerifierInterface,
                 targeter: InteractionTargeter,
                 dissolve_region: bool = False,
                 sort_criterion: Optional[str] = None):
        self.evaluator = evaluator
        self.verifier = verifier
        self.targeter = targeter
        self.dissolve_region = dissolve_region
        self.sort_criterion = sort_criterion

    def aggregate(self,
                  regions: pd.Series,
                  cores: pd.Series,
                  ) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
        """Aggregate the provided regions.

        :param regions: A mapping of unit IDs to IDs of units forming regional
            centres.
        :param cores: A boolean Series denoting which units are cores of their
            regions.
        :returns: A tuple of three objects. The first two are aggregated
            versions of regions and cores after the criteria of this aggregator
            were satisfied, and the third is the stage dataframe outlining
            at which values of the criteria the specific regions were
            aggregated.
        """
        # logging.debug('launching aggregation')
        regions = regions.copy()
        cores = cores.copy()
        evaluations = self.evaluator.eval_all(regions, cores)
        sort_crit = (
            self.sort_criterion if self.sort_criterion is not None
            else evaluations.columns[0]
        )
        stages = []
        # logging.debug('initial evaluation of %d regions', len(evaluations.index))
        # self._show_evaluations(evaluations)
        while True:
            aggreg_code = evaluations[sort_crit].idxmin()
            aggreg_eval = tuple(evaluations.loc[aggreg_code,:])
            if self.verifier.verify(*aggreg_eval):
                # this one is stable, set evaluation to infinity and continue
                # logging.debug('region %s stable at %s', aggreg_code, aggreg_eval)
                if not np.isfinite(evaluations.loc[aggreg_code,sort_crit]) or len(aggreg_eval) == 1:
                    # logging.debug('all regions stable, terminating')
                    break
                else:
                    evaluations.loc[aggreg_code, sort_crit] = np.inf
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
        # self._show_evaluations(evaluations)
        stage_df = pd.DataFrame.from_records(stages, columns=evaluations.columns)
        return regions, cores, stage_df

    @staticmethod
    def _show_evaluations(evaluations: pd.DataFrame) -> None:
        for row in evaluations.itertuples(name=None):
            print('  %s: %s', row[0], ', '.join([str(val) for val in row[1:]]))
            logging.debug('  %s: %s', row[0], ', '.join([str(val) for val in row[1:]]))

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
