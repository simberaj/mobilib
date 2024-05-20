"""Connect (cut) roads so that they only touch at endpoints."""

import os
from typing import Any, Optional, List

import geopandas as gpd

import mobilib
import mobilib.argparser
import mobilib.routing


def roads_to_network(source_file: os.PathLike,
                     target_file: os.PathLike,
                     attrs: List[str] = [],
                     oneway_col: Optional[str] = None,
                     forward_way_value: Optional[Any] = None,
                     backward_way_value: Optional[Any] = None,
                     tolerance: float = 0.,
                     largest_component_only: bool = False,
                     ) -> None:
    if attrs and oneway_col not in attrs:
        attrs.append(oneway_col)
    gdf = gpd.read_file(source_file)
    gdf = mobilib.routing.interconnect(gdf)
    network = mobilib.routing.from_lines(gdf, attrs, tolerance)
    mobilib.routing.fix_orientation(
        network, oneway_col, forward_way_value, backward_way_value
    )
    if largest_component_only:
        network = mobilib.routing.largest_component(network)
    line_gdf = mobilib.routing.to_lines(network, gdf.crs)
    line_gdf.to_file(os.fspath(target_file))


parser = mobilib.argparser.default(__doc__)
parser.add_argument(
    'in_roads',
    help='GDAL-compatible spatial file with connected (routable) roads'
)
parser.add_argument(
    'out_network',
    help='path to output a GDAL-compatible spatial file with cleaned roads'
         ' of the main network'
)
parser.add_argument(
    '-a', '--attrs', nargs='+',
    help='attributes of the roads to preserve in the network (default:'
         ' preserve all)'
)
parser.add_argument(
    '-o', '--oneway-col',
    help='oneway street attribute name, will be used to determine edge'
         ' directions and dropped'
)
parser.add_argument(
    '-f', '--forward-way-value', default='F',
    help='oneway street attribute value signifying rideability only in the'
         ' direction of geometry'
)
parser.add_argument(
    '-r', '--reverse-way-value', default='T',
    help='oneway street attribute value signifying rideability only against'
         ' the direction of geometry'
)
parser.add_argument(
    '-t', '--tolerance', type=float, default=0.001,
    help='geometry tolerance for node snapping'
)
parser.add_argument(
    '-l', '--largest-component-only',
    help='oneway street attribute name, will be used to determine edge'
         ' directions and dropped'
)


if __name__ == '__main__':
    args = parser.parse_args()
    roads_to_network(
        args.in_roads,
        args.out_network,
        args.attrs,
        args.oneway_col,
        args.forward_way_value,
        args.reverse_way_value,
        args.tolerance,
        args.largest_component_only,
    )
