"""Domain constants for the (synthetic) Dubai residential market."""
from __future__ import annotations

# ppsf = base price per sqft (AED, Jan-2022), growth = yearly appreciation,
# sea = probability of a sea view, tall = typical tower height, w = share of transactions.
AREAS: dict[str, dict] = {
    "Downtown Dubai":          dict(ppsf=2300, growth=0.14, sea=0.00, tall=45, w=1.1, villa=0.00),
    "Dubai Marina":            dict(ppsf=1900, growth=0.12, sea=0.35, tall=40, w=1.4, villa=0.00),
    "Palm Jumeirah":           dict(ppsf=3200, growth=0.18, sea=0.70, tall=15, w=0.7, villa=0.30),
    "Business Bay":            dict(ppsf=1800, growth=0.13, sea=0.00, tall=35, w=1.2, villa=0.00),
    "Jumeirah Village Circle": dict(ppsf=1100, growth=0.15, sea=0.00, tall=12, w=1.6, villa=0.05),
    "Dubai Hills Estate":      dict(ppsf=1900, growth=0.16, sea=0.00, tall=20, w=0.9, villa=0.35),
    "Arabian Ranches":         dict(ppsf=1400, growth=0.11, sea=0.00, tall=2,  w=0.6, villa=0.80),
    "Jumeirah Lake Towers":    dict(ppsf=1300, growth=0.10, sea=0.00, tall=30, w=1.0, villa=0.00),
    "Al Barsha":               dict(ppsf=1150, growth=0.09, sea=0.00, tall=10, w=0.8, villa=0.05),
    "Dubai South":             dict(ppsf=800,  growth=0.12, sea=0.00, tall=8,  w=0.7, villa=0.20),
}
PROPERTY_TYPES = ["Apartment", "Townhouse", "Villa"]
VIEWS = ["Sea", "City", "Community", "None"]
START_MONTH = "2022-01-01"
N_MONTHS = 48
