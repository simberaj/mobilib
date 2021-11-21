"""Compute geographic antipotential and store it as a raster."""

import pandas as pd

import mobilib.core
import mobilib.potential
import mobilib.argparser
import mobilib.raster

DEFAULT_HALFRANGE = 10000
DEFAULT_FULLRANGE = 400
DEFAULT_CELLSIZE = 1000

        
parser = mobilib.argparser.default(__doc__, places=True, add_places_id=False)
parser.add_argument('output_raster',
    help='path to output a GeoTIFF antipotential raster'
)
parser.add_argument('-m', '--magnitude-col',
    help='magnitude field in places with (defaults to 1 everywhere if not given)'
)
parser.add_argument('-d', '--decay-function', default='gaussian',
    help='type of distance decay function to use for potential decay'
)
parser.add_argument('-r', '--halfrange-col',
    help='field in supply points with the distance at which half of potential is gone'
)
parser.add_argument('-R', '--halfrange', default=DEFAULT_HALFRANGE, type=float,
    help='constant distance at which half of potential is gone'
)
parser.add_argument('-f', '--fullrange-col',
    help='field in supply points with the distance up to which potential is full'
)
parser.add_argument('-F', '--fullrange', default=DEFAULT_FULLRANGE, type=float,
    help='constant distance up to which potential is full'
)
parser.add_argument('-s', '--cell-size', default=DEFAULT_CELLSIZE, type=float,
    help='cell size of the output antipotential raster'
)
parser.add_argument('-C', '--output-srid', default=None, type=int,
    help='EPSG SRID for output raster, should be a projected system; defaults to match input'
)

def prepare_sources(places_gdf,
                    magnitude_col=None,
                    halfrange_col=None,
                    fullrange_col=None,
                    halfrange=DEFAULT_HALFRANGE,
                    fullrange=DEFAULT_FULLRANGE,
                    output_srid=None,
                    **kwargs):
    if output_srid is not None:
        places_gdf = places_gdf.to_crs(output_srid)
    sources_df = pd.DataFrame({
        'x': places_gdf.geometry.x,
        'y': places_gdf.geometry.y,
    })
    transfer_column(places_gdf, sources_df, magnitude_col, 'magnitude', 1)
    transfer_column(places_gdf, sources_df, halfrange_col, 'halfrange', halfrange)
    transfer_column(places_gdf, sources_df, fullrange_col, 'fullrange', fullrange)
    return sources_df


def transfer_column(src_df, tgt_df, src_col, tgt_col, default):
    if src_col is None or src_col not in src_df.columns:
        tgt_df[tgt_col] = default
    else:
        tgt_df[tgt_col] = src_df[src_col]


if __name__ == '__main__':
    args = parser.parse_args()
    sources_df = prepare_sources(mobilib.core.read_places(args), **vars(args))
    potential_arr, world = mobilib.potential.raster(
        sources_df,
        args.cell_size,
        decay=args.decay_function
    )
    output_srid = int(args.output_srid if args.output_srid else args.srid)
    mobilib.raster.to_geotiff(
        potential_arr, args.output_raster, world, srid=output_srid
    )
