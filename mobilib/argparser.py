"""Precomposed argument parsers for this package's scripts."""

import argparse

def default(docstring: str,
            interactions: bool = False,
            areas: bool = False,
            places: bool = False,
            add_places_id: bool = True,
            add_interaction_strength: bool = True,
            ):
    """Create a default argument parser with docstring as main help.

    Optionally, also add some common groups of options.
    """
    parser = argparse.ArgumentParser(
        description=docstring,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    if interactions: add_interactions(parser, add_strength=add_interaction_strength)
    if areas: add_areas(parser)
    if places: add_places(parser, add_id=add_places_id)
    return parser


def add_interactions(parser, add_strength=True):
    parser.add_argument('inter_file',
        help='interaction data as a semicolon-delimited CSV'
    )
    parser.add_argument('-f', '--from-id-col', default='from_id',
        help='name of the source area/place ID attribute in the interactions file'
    )
    parser.add_argument('-t', '--to-id-col', default='to_id',
        help='name of the target area/place ID attribute in the interactions file'
    )
    if add_strength:
        parser.add_argument('-s', '--strength-col', default='strength',
            help='name of the interaction strength attribute in the interactions file'
        )


def add_areas(parser):
    parser.add_argument('area_file',
        help='interacting areas layer as a GDAL-compatible polygon file'
    )
    parser.add_argument('-i', '--area-id-col', default='id',
        help='name of the area ID attribute in the area file'
    )


def add_places(parser, add_id: bool = True):
    parser.add_argument('place_file',
        help='interacting places layer as a GDAL-compatible point file or CSV'
    )
    if add_id:
        parser.add_argument('-i', '--place-id-col', default='id',
            help='name of the place ID attribute in the place file'
        )
    parser.add_argument('-x', '--x-col', default='X',
        help='name of the x-coordinate attribute in the place file (for CSV)'
    )
    parser.add_argument('-y', '--y-col', default='Y',
        help='name of the y-coordinate attribute in the place file (for CSV)'
    )
    parser.add_argument('-c', '--srid', default=4326,
        help='EPSG SRID of the place coordinates (for CSV)'
    )
