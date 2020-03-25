'''Delimit functional regions based on spatial interactions.

Aggregate spatial units step-wise until all their regions satisfy the specified
verification criteria, such as population size.
'''

import sys
import logging

import pandas as pd
import mobilib.argparser
import mobilib.region

parser = mobilib.argparser.default(__doc__, interactions=True)
parser.add_argument('-u', '--unit-file',
    help='interacting unit data as a semicolon-delimited CSV'
)
parser.add_argument('-i', '--unit-id-col', default='id',
    help='name of the unit ID attribute in the unit file'
)
parser.add_argument('-p', '--unit-prop-col', nargs=2, action='append', default=[],
    metavar=('PROPERTY', 'COLUMN'),
    help='name of the unit property and its corresponding column in the unit file to load and use'
)
parser.add_argument('-C', '--core-preset-col', default='multicore',
    help='name of the unit preset multi-core regional assignment column in the unit file'
)
parser.add_argument('-R', '--hinterland-preset-col', default='region',
    help='name of the unit preset regional assignment column in the unit file'
)
parser.add_argument('-c', '--verify-criterion',
    nargs=2, action='append', default=[],
    metavar=('CRITERION', 'VALUE'),
    help='criterion name and threshold for all regions to be delimited'
)
parser.add_argument('-d', '--dissolve-agg-region', action='store_true',
    help='reassign aggregated regions by individual unit, not as a whole'
)
parser.add_argument('-v', '--verbose', action='store_true',
    help='show detailed progress messages'
)
parser.add_argument('out_file',
    help='path to output the regional assignments as a semicolon-delimited CSV'
)

def get_unit_props(unit_df, prop_defs):
    if unit_df is None:
        return {}
    else:
        return {
            prop_name: unit_df[prop_col]
            for prop_name, prop_col in prop_defs
        }
    
def apply_region_presets(regions, cores, presets, core=False):
    presets = presets.dropna()
    if pd.api.types.is_numeric_dtype(presets) and (presets.astype(int) == presets).all():
        presets = presets.astype(int)
    n_presets = len(presets.index)
    if n_presets:
        logging.debug('applying %d assignment presets (core: %s)', n_presets, core)
        regions[presets.index] = presets
        if not core:
            cores[presets.index] = False


def load_interactions(args):
    logging.debug('loading interactions from %s', args.inter_file)
    inter = pd.read_csv(args.inter_file, sep=';').rename(columns={
        args.from_id_col: 'from_id',
        args.to_id_col: 'to_id',
    }).set_index(['from_id', 'to_id'])[args.strength_col]
    logging.debug('%d interactions loaded', len(inter.index))
    return inter


def load_units(args):
    logging.debug('loading units from %s', args.unit_file)
    unit_df = pd.read_csv(args.unit_file, sep=';').set_index(args.unit_id_col)
    logging.debug('%d units loaded', len(unit_df.index))
    return unit_df


def output(unit_df, regions, cores, out_path):
    if unit_df is None:
        unit_df = pd.DataFrame()
    unit_df['region'] = regions
    unit_df['is_core'] = cores
    if unit_df.index.name is None:
        unit_df.index.name = 'unit_id'
    logging.debug('saving output to %s', out_path)
    unit_df.to_csv(out_path, sep=';')


def create_evaluator(args):
    evaluator = mobilib.region.evaluator(
        [criterion for criterion, threshold in args.verify_criterion]
    )
    for criterion in evaluator.get_required_properties():
        if not any(property == criterion for property, column in args.unit_prop_col):
            raise ValueError(f'unit property-based criterion {criterion}'
                              ' specified but not found in unit properties')
    return evaluator


def create_verifier(args):
    partials = [
        mobilib.region.MinimumVerifier(float(threshold))
        for criterion, threshold in args.verify_criterion
    ]
    if not partials:
        return mobilib.region.YesmanVerifier()
    elif len(partials) == 1:
        return partials[0]
    else:
        return mobilib.region.CompoundVerifier(partials)


def create_targeter(args):
    return mobilib.region.InteractionTargeter(
        source_core=True,
        target_core=True,
    )

def create_aggregator(args, evaluator, verifier, targeter):
    return mobilib.region.StepwiseAggregator(
        evaluator=evaluator,
        verifier=verifier,
        targeter=targeter,
        dissolve_region=args.dissolve_agg_region
    )

import warnings
warnings.filterwarnings("error")

if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(
        stream=sys.stdout,
        level=(logging.DEBUG if args.verbose else logging.INFO),
        format='%(message)s',
    )
    evaluator = create_evaluator(args)
    verifier = create_verifier(args)
    targeter = create_targeter(args)
    aggregator = create_aggregator(args, evaluator, verifier, targeter)
    inter = load_interactions(args)
    unit_id_set = frozenset(inter.index.get_level_values('from_id'))
    logging.debug('%d units identified in interactions', len(unit_id_set))
    if args.unit_file:
        unit_df = load_units(args)
        unit_id_set |= frozenset(unit_df.index.tolist())
    else:
        unit_df = None
    unit_props = get_unit_props(unit_df, args.unit_prop_col)
    unit_ids = list(sorted(unit_id_set))
    logging.debug('%d total units identified', len(unit_ids))
    # initial setup: each unit is a core of its own region
    regions = pd.Series(unit_ids, index=unit_ids)
    cores = pd.Series(True, index=unit_ids)
    # apply presets where they are specified
    for col, core in ((args.core_preset_col, True), (args.hinterland_preset_col, False)):
        if col and unit_df is not None and col in unit_df.columns:
            apply_region_presets(regions, cores, unit_df[col], core=core)
    evaluator.feed(inter, unit_props)
    targeter.feed(inter, unit_props)
    agg_regions, agg_cores = aggregator.aggregate(regions, cores)
    
    # other missing:
    # aggregation ordering setting (by ascending aggregation/descending verification/ascending verification)
    # flow transformations
    # aggregation ordering for partitioned dissolved region
    # neighbourhood, exclave determination and elimination
    
    # to test:
    # target core/region
    # source from core/region
    # partition/not to partition dissolved region
    
    # RUN AGGREGATION
    # different verification criteria based on unit properties, interaction measures...
    
    output(unit_df, agg_regions, agg_cores, args.out_file)
