
'''Creates regions by hierarchy aggregation using a minimum weight threshold.'''

import argparse

import pandas as pd

import mobilib.hierarchy

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('hierarchy_file',
    help='semicolon-delimited CSV file with hierarchy definitions')
parser.add_argument('out_file', help='file to write the output regional assignment')
parser.add_argument('-i', '--id-col', default='id',
    help='unit ID column in hierarchy file')
parser.add_argument('-p', '--parent-col', default='parent_id',
    help='parent unit ID column in hierarchy file')
parser.add_argument('-o', '--organic-col', default='organ',
    help='organic assignment boolean column in hierarchy file')
parser.add_argument('-w', '--weight-col', default='weight',
    help='unit weight column in hierarchy file')
parser.add_argument('-r', '--region-col', default='region',
    help='name of the aggregated region column in the output file')
parser.add_argument('-m', '--min-weight', default=None, type=float,
    help='minimum weight of region; if not set, determine the optimal one from data by finding the most significant gap')
parser.add_argument('-s', '--min-weight-sel-opts', default=0, type=int,
    help='let the user select the optimal weight from this many options; only has effect when -m is not set')


def nicefy(low, high):
    mean = (low + high) * 0.5
    nplaces = int(numpy.ceil(numpy.log10(mean)))
    roundmean = numpy.around(mean, -nplaces)
    while roundmean > high or roundmean < low:
        nplaces -= 1
        roundmean = numpy.around(mean, -nplaces)
    return roundmean


if __name__ == '__main__':
    args = parser.parse_args()
    hierdf = pd.read_csv(args.hierarchy_file, sep=';')
    ids = hierdf[args.id_col]
    weights = hierdf[args.weight_col].values
    hierarchy = mobilib.hierarchy.Hierarchy.create(
        parents=hierdf[args.parent_col],
        organics=hierdf[args.organic_col],
        ids=ids,
        weights=weights,
        parents_as_ids=True,
    )
    crclass = mobilib.hierarchy.WeightAggregationCriterion
    if args.min_weight:
        criterion = crclass(minimum=args.min_weight)
    else:
        criterion = crclass.optimal(hierarchy, weights, select=args.min_weight_sel_opts)
        if not args.min_weight_sel_opts:
            print('Optimal weight threshold determined at {:n}'.format(criterion.minimum))
    regsys = criterion.aggregate(hierarchy, weights)
    hierdf[args.region_col] = ids[regsys.to_array()].values
    hierdf.to_csv(args.out_file, sep=';', index=False)
    
    # import numpy
    # levels = numpy.sort(criterion.get_levels(hierarchy, weights))
    # # print(levels.astype(int))
    # n = levels.size
    # ratings = numpy.diff(levels) / levels[1:] * numpy.log(n - numpy.arange(n-1) - 1)
    # start = n // 2
    # best_splits = numpy.argsort(ratings[start:])[-8:] + start
    # # print(best_splits)
    # for i in reversed(best_splits):
        # print(i, n-i-1, levels[i], nicefy(*levels[i:i+2]))
    # bounds = [nicefy(*item) for item in zip(levels[best_splits], levels[best_splits+1])]
    # print(ratings[best_splits])
    # print(bounds)
    # choice based on ratio of smallest retained region size to the eliminated region size

    # print(levs.astype(int))
    # open('hier.txt', 'w', encoding='utf8').write(hierarchy.structure_string())
    # regsys = criterion.aggregate(hierarchy, weights)
    # reg_indices = regsys.to_array()
    # print(reg_indices)
    # hierdf[args.region_col] = [ids[i] for i in reg_indices]
    # hierdf.to_csv(args.out_file, sep=';', index=False)

    # print(regsys)
    # print(regsys.to_array())
    # def create(cls, parents, organics=None, ids=None, root=None, rels=None):

    # rels, ids = mobilib.hierarchy.Relations.from_dataframe(
        # reldf, from_col, to_col, strength_col
    # )
    # builder = mobilib.hierarchy.MaxflowHierarchyBuilder()
    # # builder = mobilib.region.GeneticHierarchyBuilder()
    # hierarchy = builder.build(rels, ids=ids)
    # # print(hierarchy.structure_string())
    # # criterion = mobilib.hierarchy.TransitionCriterion()
    # # mover = mobilib.hierarchy.RootMover()
    # # mover.modify(hierarchy, rels)
    # parents, organics = hierarchy.to_arrays()
    # outdf = pd.DataFrame({
        # 'id' : ids,
        # 'parent_id' : [ids[parent] for parent in parents],
        # 'organ' : organics
    # })
    # if args.save_weights:
        # outdf['weight'] = rels.weights
    # if args.place_file:
        # outdf = pd.merge(
            # pd.read_csv(args.place_file, sep=';'),
            # outdf,
            # how='outer',
            # left_on=args.place_id_col,
            # right_on='id',
        # )
    # outdf.to_csv(args.out_file, sep=';', index=False)
