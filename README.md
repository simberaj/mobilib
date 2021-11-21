# mobilib: a Python suite for mobile positioning data in urban system modeling
This library and script collection encompasses a wide variety of tooling that I
developed during my dissertation on mobile positioning data processing for
urban system modeling.

Some of the scripts belong to workflows presented in one of my articles; those
are grouped under their thematic descriptions. Other scripts and utilities
are described further below. 


## Commuting phones
>   Šimbera, J., Aasa, A.: Commuting phones: Moving on from enumeration
    to mobile positioning in commuting statistics. Journal of Urban Technology
    (in review).

This article shows how to convert mobile positioning-derived anchor points of
individual people to aggregate spatial interactions (commuting flows), which
are of great importance in spatial analysis.

The main body of the code is located in the `mobilib.relations` module.
Run the scripts as follows:

-   `anchors_to_rels` to get interactions from anchor points
    -   `calculate_eta` might be used to calculate the self-interaction
        parameter eta, but it does not really help with anything
-   `transfer_rels` to areally interpolate interactions between two sets of
    spatial units (such as from mobile network cells to administrative units)
    -   `transfer_table` to create the required transfer table that specifies
        the fractions of source spatial units (mobile network cells) to be
        transferred to respective target spatial units (administrative units)
        using a weighting layer that specifies how the users are distributed
        within the source spatial units
    -   `transfer_table_aw` to substitute `transfer_table` by a simpler areal
        weighting process that is less accurate but does not require the
        weighting layer
-   `calibrate_rels` to calibrate interaction values to census or other
    calibration data
-   `rels_to_lines` to visualize the created interactions into a line
    geometry spatial file
    -   `representative_points` converts a spatial file to a point file
        of representative points as provided by the Shapely algorithm (useful
        if you only have a polygon file to represent interacting units)
    -   `weighted_centroids` can be used to do the same but in a more advanced
        fashion, using a weighting layer (such as the layer of built-up areas)

## Hierarchical settlement system model
>   Šimbera, J., Aasa, A.: Hierarchical settlement system model: making
    polycentricity work with central place theory. Urban Studies (in review).

This article presents a way to model relationships and hierarchies
in the settlement system using a model with three developmental stages.
Here is how to create such a model from spatial interactions, evaluate and
visualize it.

The main body of the code is located in the `mobilib.hssm` module.
Run the scripts as follows:

-   `build_hssm` to build the hierarchical settlement system model and store it
    in a table
-   `hssm_to_regions` to transform the model into a set of functional regions
    with defined criteria (not currently operational)
    -   `show_zipf_gaps` may reveal which population criteria might be the most
        meaningful for the given settlement system


## Functional region delimitation and measurement
>   Komárek, M., Marada, M., Šimbera, J.: Metropolitní areály v Česku.
    (work in progress)

The main body of the code is located in `mobilib.region` where regions can be
formed by aggregation, and in `mobilib.region_measure` where arbitrary
groupings of units can be measured based on functional region criteria.
There are the following scripts:

-   `add_distances` to make interactions distance-aware
-   `flow_intensity_indices` to calculate some interaction metrics such as
    counts of commuters per kilometer of distance
-   `delimit_multicores` to find groupings of units that form a core of a
    common region
-   `delimit_regions` which annotates units by region ID they belong to, based
    on a sequential (CURDS-like) functional region delimitation procedure
    -   `show_zipf_gaps` may also help here (see above)
-   `measure_regions` to compute various metrics on such regions
-   `summarize_unit_rels` to compute metrics on units based on their regional
    assignment
-   `dissolve_areas` to turn these region annotations and unit geometries
    into region polygons with proper attributes
-   `eliminate_exclaves` to eliminate exclaves of these region polygons by
    merging them with neighboring regions


## Other utilities
-   `calibrate` and `calibrate_batch` are counterparts of `calibrate_rels`
    for non-interaction data such as population counts.
-   `equalize_polygons` makes a polygon set approximate a given uniform surface
    area by aggregation or splitting; this was meant as an extension of the
    HSSM study to compensate for MAUP
-   `eval_regression_error` evaluates the regression error in given data files
    by producing HTML reports
-   `fix_polygon_neighbours` fixes geometries in a polygon dataset so that
    their boundaries align nicely
-   `csv_to_shp` and `geo_to_csv` convert between GDAL-compatible spatial
    files and custom semicolon-delimited CSVs with WKT or XY geometry
-   `lines_to_network` prepares a road/rail/whatever line dataset to be
    routable using networkx
-   `raster_to_points` converts a raster to a point layer
-   `signaling_trajectories` smoothes out a point layer of trajectories
    for better inspection
-   `transfer_static` is a variant of `transfer_rels` for non-interaction data
    (keyed by one identifier, e.g. mobile network cell populations)
-   `triangulate` creates a neighbourhood interaction table of points
    by calculating which pairs of them are connected by an edge in the Delaunay
    triangulation
-   `voronoi_cells` calculates ordinary Voronoi cells for a point dataset
-   `potential_raster`, `mobilib.potential` and `mobilib.raster` contain
    old machinery to compute raster maps of geographic potential

## Obsolete stuff
-   `antenna_centroids` and `antenna_model` for mobile network antenna coverage
    modeling (together with `mobilib.antenna`): experiment not finished,
    leaving here for future reference
