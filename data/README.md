# Data

This public repository does **not** include original ABI shipment data.

## Included

- `demo_synthetic_shipments.csv`: a small synthetic dataset created only to demonstrate the pipeline, scripts, tests, and visualizations.

## Not Included

- original transaction-level shipment files
- training and test Excel workbooks from the dissertation directory
- any proprietary operational data

## Expected Schema For Private Local Data

If you want to run the code on your own local copy of private data, place a CSV file outside version control and use the following columns:

- `shipment_id`: unique shipment identifier
- `ship_date`: shipment date in `YYYY-MM-DD` format
- `distance_miles`: delivery distance
- `tractor_weight_lbs`: tractor weight
- `carrier_group`: optional categorical carrier label

## Notes

- Distances are assumed to be known before trailer loading decisions.
- Tractor weights are observed at dispatch or arrival depending on the experiment.
- Any private file should stay in a git-ignored location such as `data/private/`.
