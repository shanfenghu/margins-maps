# Public inputs

The evaluation expects two local directories under `data/raw/`.

`usgs/` contains Connecticut county and hydrologic-unit boundaries and the
associated 1985 water-use totals distributed by the US Geological Survey.
Expected names are `ct85co.txt`, `ct85w8.txt`, `ct_counties.geojson`,
`ct_counties_2010.geojson`, and `ct_huc8.geojson`.

`usda/` contains 2022 Census of Agriculture totals and public boundary files.
Expected names are `NASSAgCensusDownload2022_175maps_Links.xlsx`,
`county_farms_2022.csv`, `huc6_farms_2022.csv`,
`counties_ne_ny_2020.geojson.gz`, and `huc6_01.geojson.gz`.

These inputs are excluded from version control because they are redistributed
by their source agencies and can be downloaded independently. The workflow
checks identifiers, balance conditions, and geometry validity before analysis.
