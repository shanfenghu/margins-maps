# Public inputs

All inputs required by the workflow are included under `data/input/`. They are
small, immutable extracts from United States government sources. The expected
SHA-256 digests live in `margins_maps.inputs`; run `make check-data` before a
full evaluation.

## Water-use data

`input/usgs/` contains Connecticut county totals, hydrologic-unit totals, 2010
county boundaries, and Watershed Boundary Dataset geometries. The historical
totals originate from the [USGS 1985 water-use release](https://water.usgs.gov/watuse/data/1985/index.html).
The geometries originate from [Census TIGERweb](https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/tigerWMS_Census2010/MapServer)
and the [USGS Watershed Boundary Dataset](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset).

## Agricultural data

`input/usda/` contains 2022 farm counts on county and HUC6 supports, plus the
corresponding public boundaries. The counts originate from the
[USDA Census of Agriculture](https://www.nass.usda.gov/AgCensus/) county and
watershed resources.

The workflow validates file digests first, then checks identifiers, nonnegative
values, marginal balance, and geometry validity during evaluation.
