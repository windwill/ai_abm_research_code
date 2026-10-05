# Empirical Calibration Data Sources

Access date: 2026-05-07

This file documents the empirical inputs used to produce the generated calibration target file.
The target CSV remains editable, but each included value should retain a source note before reported calibration results are produced.

Generated outputs:

- `data\empirical\processed\empirical_annual_series.csv`
- `data\empirical\processed\calibration_targets_empirical.csv`

## Sources

### Civilian unemployment rate

- URL: https://fred.stlouisfed.org/series/UNRATE
- Status: used
- Notes: Downloaded as FRED CSV series UNRATE.

### Real gross domestic product

- URL: https://fred.stlouisfed.org/series/GDPC1
- Status: used
- Notes: Downloaded as FRED CSV series GDPC1.

### Federal debt held by the public as percent of GDP

- URL: https://fred.stlouisfed.org/series/GFDEGDQ188S
- Status: used
- Notes: Downloaded as FRED CSV series GFDEGDQ188S.

### Average hourly earnings of all private employees

- URL: https://fred.stlouisfed.org/series/CES0500000003
- Status: used
- Notes: Downloaded as FRED CSV series CES0500000003.

### Consumer price index for all urban consumers

- URL: https://fred.stlouisfed.org/series/CPIAUCSL
- Status: used
- Notes: Downloaded as FRED CSV series CPIAUCSL.

### ACS national income Gini

- URL: https://api.census.gov/data/2024/acs/acs1?get=NAME,B19083_001E&for=us:1
- Status: used
- Notes: Latest available ACS 1-year estimate.

### BDS firm exits

- URL: https://api.census.gov/data/timeseries/bds
- Status: used
- Notes: Firm deaths divided by firm counts.

### AI adoption point estimate

- URL: https://www.federalreserve.gov/econres/notes/feds-notes/monitoring-ai-adoption-in-the-u-s-economy-20260403.html
- Status: used
- Notes: Federal Reserve note reports Census BTOS firm-level AI adoption at about 18 percent at year-end 2025.

## Processing Notes

- FRED unemployment is converted from percent to a rate for `U_mean`.
- Real GDP is converted to annual log growth for `Y_growth`.
- Federal debt is converted from percent of GDP to a ratio for `Debt_to_GDP_final`.
- Average hourly earnings are divided by CPI and normalized to an index for validation rather than direct calibration.
- ACS income Gini is treated as a proxy for the model inequality statistic.
- The AI adoption target is a documented point estimate from a public Federal Reserve summary of Census BTOS rather than a direct API pull.
