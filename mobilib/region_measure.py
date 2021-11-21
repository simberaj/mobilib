"""Measure arbitrary groupings of units based on functional region criteria."""

import inspect
import re
from typing import List, Dict, Optional

import numpy as np
import pandas as pd


TYPE_PREFIX: str = '_is_type_'


def _get_type_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col.startswith(TYPE_PREFIX)]


def _get_available_types(df: pd.DataFrame) -> List[str]:
    return [col[len(TYPE_PREFIX):] for col in df.columns if col.startswith(TYPE_PREFIX)]


# AUXILIARY SOURCES

def labeled_units(units: pd.DataFrame, largest_prop: str, largest_fraction: float) -> pd.DataFrame:
    label_df = pd.DataFrame([], index=units.index)
    if largest_prop in units.columns:
        grouped_largest = units[largest_prop].groupby(units['region'])
        if largest_fraction < 1:
            is_largest = units.join(
                grouped_largest.max().rename('_maxsize'), on='region'
            ).eval(f'{largest_prop} >= {largest_fraction} * _maxsize')
        else:
            is_largest = units.index.isin(grouped_largest.idxmax())
        label_df[TYPE_PREFIX + 'largest'] = is_largest
    if 'is_core' in units.columns:
        label_df[TYPE_PREFIX + 'core'] = units['is_core']
        label_df[TYPE_PREFIX + 'hinterland'] = ~units['is_core']
    return units.join(label_df)


def labeled_interactions(interactions: pd.DataFrame, labeled_units: pd.DataFrame) -> pd.DataFrame:
    narrow_units = labeled_units[['region'] + _get_type_columns(labeled_units)]
    id_from, id_to = interactions.index.names
    return interactions.join(narrow_units.add_suffix('_from'), on=id_from, how='left')\
                       .join(narrow_units.add_suffix('_to'), on=id_to, how='left')


def names(units: pd.DataFrame, largest_prop: str, largest_fraction: float) -> pd.DataFrame:
    if 'name' not in units.columns:
        return pd.DataFrame()
    priority_prop = largest_prop if largest_prop in units.columns else 'name'
    print(largest_prop, largest_fraction)
    if largest_prop in units.columns and largest_fraction < 1:
        is_name_source = units.join(
            units[priority_prop].groupby(units['region']).max().rename('_maxsize'),
            on='region'
        ).eval(f'{priority_prop} >= {largest_fraction} * _maxsize')
    elif 'is_core' in units.columns:
        is_name_source = units['is_core']
    else:
        is_name_source = None
    units = units.sort_values(priority_prop, ascending=False)
    names = pd.Series(np.nan, index=units['region'].unique(), name='name')
    if is_name_source is not None:
        core_names = units[is_name_source].groupby('region')['name'].agg('–'.join)
        names.fillna(core_names, inplace=True)
    if names.isna().any():
        largest_names = units.loc[units[priority_prop].groupby(units['region']).idxmax(), 'name']
        names.fillna(largest_names, inplace=True)
    return pd.DataFrame(names)


def unit_prop_sums(labeled_units: pd.DataFrame) -> pd.DataFrame:
    """Sum given properties from units to region.

    Sums the properties both in total and by unit type (absolute and relative).
    """
    # Also count units implicitly.
    if 'unit_count' not in labeled_units.columns:
        labeled_units = labeled_units.assign(unit_count=1)
    props = [
        col for col in labeled_units.columns
        if col not in ('region', 'is_core', 'name') and not col.startswith(TYPE_PREFIX)
    ]
    
    def _compute_stats(df, suffix):
        return df.groupby('region')[props].sum().add_suffix('_' + suffix)
    
    # Compute stats for entire regions.
    by_region = _compute_stats(labeled_units, 'region')
    type_dfs = []
    for unit_type in _get_available_types(labeled_units):
        # Select only units of the given type (by binary flag column) and compute the same.
        by_type = _compute_stats(labeled_units[labeled_units[TYPE_PREFIX + unit_type]], unit_type)
        # Join all-region data and compute fractions.
        by_type_merged = by_type.join(by_region)
        for prop in props:
            by_type[f'{prop}_{unit_type}_frac'] = by_type_merged.eval(
                f'{prop}_{unit_type} / {prop}_region'
            )
        type_dfs.append(by_type)
    # Join everything together.
    measures = by_region.join(type_dfs).fillna(0).drop(
        ['unit_count_largest'], axis=1, errors='ignore'
    )
    # Retype some columns back to int in case NAs caused them to switch to float.
    for type_df in type_dfs + [by_region]:
        for col in type_df.columns:
            if col in measures and not np.issubdtype(measures[col].dtype, type_df[col].dtype):
                measures[col] = measures[col].astype(type_df[col].dtype)
    return measures


def _sum_interactions(df: pd.DataFrame, key: str = 'region_from') -> pd.Series:
    return df.groupby(key)[df.columns[0]].sum()


def _intersum_measure_name(from_type: str, to_type: str) -> str:
    from_disp = from_type if from_type != 'any' else 'region'
    to_disp = to_type if to_type != 'any' else 'region'
    return f'inter_{from_disp}_to_{to_disp}'


def interaction_sums(labeled_interactions: pd.DataFrame) -> pd.DataFrame:
    """Sum interaction strengths by type across the region."""
    within_region = labeled_interactions.eval('region_from == region_to')
    across_regions = ~within_region
    labeled_interactions = labeled_interactions.assign(**{
        TYPE_PREFIX + 'any_from': True,
        TYPE_PREFIX + 'any_to': True,
    })
    unit_types = [
        utype[:-5] if utype.endswith('_from') else utype[:-3]
        for utype in _get_available_types(labeled_interactions)
    ]
    regions = np.sort(np.array(list(set(
        labeled_interactions['region_from'].unique().tolist()
        + labeled_interactions['region_to'].unique().tolist()
    ))))
    measures = pd.DataFrame([], index=regions)
    for from_unit_type in unit_types:
        for to_unit_type in unit_types:
            measure_name = _intersum_measure_name(from_unit_type, to_unit_type)
            measures[measure_name] = _sum_interactions(labeled_interactions[
                labeled_interactions[f'{TYPE_PREFIX}{from_unit_type}_from']
                & labeled_interactions[f'{TYPE_PREFIX}{to_unit_type}_to']
                & within_region
            ]).reindex(regions, fill_value=0)
    for unit_type in unit_types:
        measures[_intersum_measure_name(unit_type, 'out')] = _sum_interactions(labeled_interactions[
            labeled_interactions[f'{TYPE_PREFIX}{unit_type}_from'] & across_regions
        ]).reindex(regions, fill_value=0)
        measures[_intersum_measure_name(unit_type, 'all')] = _sum_interactions(labeled_interactions[
            labeled_interactions[f'{TYPE_PREFIX}{unit_type}_from']
        ]).reindex(regions, fill_value=0)
        measures[_intersum_measure_name('out', unit_type)] = _sum_interactions(labeled_interactions[
            labeled_interactions[f'{TYPE_PREFIX}{unit_type}_to'] & across_regions
        ], key='region_to').reindex(regions, fill_value=0)
        measures[_intersum_measure_name('all', unit_type)] = _sum_interactions(labeled_interactions[
            labeled_interactions[f'{TYPE_PREFIX}{unit_type}_to']
        ], key='region_to').reindex(regions, fill_value=0)
    return measures


# def membership_functions(labeled_interactions: pd.DataFrame, labeled_units:
# could be added: membership functions: weighted fuzzy, weighted,
# name: memfx_<memtype>_<fuzztype>, memmass_(...)
# Hampl membership function value (U-C / (U-C + U-O))
# Feng membership function value (U-R / (U-R + U-O))


INTEGRITY_EXPRS: Dict[str, str] = {
    'hampl_hinterland_integrity': 'inter_hinterland_to_core / inter_hinterland_to_out',
    'hampl_region_integrity': '(inter_hinterland_to_core + inter_core_to_hinterland) / inter_region_to_out',
    'bezak_integrity': 'inter_region_to_region / (inter_region_to_out + inter_out_to_region)',
    'coombes_residence_integrity': 'inter_region_to_region / inter_region_to_out',
    'coombes_workplace_integrity': 'inter_region_to_region / inter_out_to_region',
    'coombes_self_containment': '(inter_region_to_region / inter_region_to_out + inter_region_to_region / inter_out_to_region) / 2',
}


def integrities(interaction_sums: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        name: interaction_sums.eval(expr) for name, expr in INTEGRITY_EXPRS.items()
    })


INTER_OPTS: str = 'largest|core|hinterland|region|out|all'
INTER_REGEX: str = fr'inter_({INTER_OPTS})_to_({INTER_OPTS})'

MEASURES: Dict[str, str] = {
    'names': 'name',
    'unit_prop_sums': fr'(?!{INTER_REGEX})\w+_(region|((largest|core|hinterland)(_frac)?))',
    'interaction_sums': INTER_REGEX,
    'integrities': '|'.join(INTEGRITY_EXPRS.keys()),
}

INTERMEDIARIES: List[str] = ['labeled_units', 'labeled_interactions']

MEASURE_FXS = {name: globals()[name] for name in list(MEASURES) + INTERMEDIARIES}


def calculate(units: pd.DataFrame,
              interactions: pd.DataFrame,
              only_measures: Optional[List[str]] = None,
              exclude_measures: List[str] = [],
              largest_prop: str = 'mass',
              largest_fraction: float = 1.,
              ) -> pd.DataFrame:
    """Calculate numerical measures for regions given by the spatial units and their interactions.

    :param units: A dataset of units constituting the regions. It must contain a ``region`` column
        with region identifiers. It may contain a ``name`` column with the unit names to be
        assembled to region names, a ``is_core`` column with a binary flag whether the unit is
        part of the core of its region, and an arbitrary number of numerical property columns
        to be summarized.
    :param interactions: A dataset of interactions between the units. It must have a two-level
        index whose values reference the index of the units dataframe. Its first column is assumed
        to contain interaction strength, other columns are disregarded.
    :param only_measures: Only return the selected measures.
    :param exclude_measures: Return all but the specified measures.
    :param largest_prop: Name of one of the unit property columns that is taken as a measure of
        unit size for the purposes of determining the largest unit of the region.
    :param largest_fraction: The fraction of largest unit size that makes a unit to also be
        considered largest. (If this is less than one, region names are only derived from
        largest units and not all core units.)
    :returns: A dataframe with region measures, one column per each, with region identifiers from
        the units dataframe as its index.
    """
    if only_measures is None:
        end_measurers = [MEASURE_FXS[name] for name in MEASURES]
    else:
        end_measurers = []
        for name, measure_regex in MEASURES.items():
            if any(re.fullmatch(measure_regex, m) for m in only_measures):
                end_measurers.append(MEASURE_FXS[name])
    compute_stack = end_measurers[:]
    data = {
        'units': units,
        'interactions': interactions,
        'largest_prop': largest_prop,
        'largest_fraction': largest_fraction,
    }
    while compute_stack:
        for fx in compute_stack:
            fx_args = list(inspect.signature(fx).parameters.keys())
            missing_args = [arg for arg in fx_args if arg not in data]
            if missing_args:
                # Add step prerequisites to the compute stack.
                for arg in missing_args:
                    dep_fx = globals()[arg]
                    if dep_fx not in compute_stack:
                        print('adding prerequisite', arg)
                        compute_stack.append(dep_fx)
            else:
                # Compute the step if all inputs are ready.
                step_name = fx.__name__
                print('computing', step_name)
                data[step_name] = fx(**{arg: data[arg] for arg in fx_args})
                compute_stack.remove(fx)
    outputs = [data[endpoint.__name__] for endpoint in end_measurers]
    output_df = outputs[0]
    if len(outputs) > 1:
        output_df = output_df.join(outputs[1:])
    if only_measures is not None:
        output_df = output_df[only_measures]
    if exclude_measures:
        output_df.drop(exclude_measures, axis=1, errors='ignore', inplace=True)
    return output_df
