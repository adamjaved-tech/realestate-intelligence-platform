# Valuation methodology

The valuation model is a gradient-boosted tree regressor (LightGBM) trained on the logarithm of the transaction price. Predicting the log price makes errors proportional to the price, so a 5% miss on a studio and a 5% miss on a villa are treated alike.

Each valuation returns a point estimate and an 80% prediction interval. The interval comes from two additional quantile models trained at the 10th and 90th percentiles. If the interval coverage on held-out data drifts far from 80%, the promotion gate blocks the model from becoming champion.

Market features describe the neighbourhood at the time of sale: the median price per sqft of the previous month, the average of the previous three months, the previous month's transaction count and the month-on-month growth. They only use information from months before the transaction month, which prevents target leakage.

The model is evaluated on the most recent six months of data that it never saw during training or early stopping. A naive baseline of size multiplied by the previous month's area median price per sqft is reported next to every model, and the model must beat it to be promoted.

# Data quality rules

Raw transactions land in the bronze layer untouched. The silver layer removes exact duplicate transaction identifiers, standardises area and property type names, and quarantines rows with invalid dates, unknown areas, non-positive prices, missing sizes, implausible bedroom counts or a price per sqft outside 200 to 15,000 AED. Quarantined rows are kept with a reject reason for audit. If more than 10% of rows are rejected, the pipeline fails instead of publishing bad data.

# Monitoring

Data drift is measured with the Population Stability Index. A PSI below 0.10 is stable, 0.10 to 0.25 is a warning, and above 0.25 is an alert. A Kolmogorov-Smirnov test is reported for numeric features. Drift is tracked on property-level inputs and on the ratio between the model prediction and the naive baseline, not on market-level features, which change every month by design.

# Glossary

Price per sqft is the transaction price in AED divided by the built-up area in square feet. A deal alert is raised when an asking price is more than 15% below the model fair value, which suggests an undervalued listing, or more than 10% above the upper bound of the interval, which suggests an overpriced listing.
