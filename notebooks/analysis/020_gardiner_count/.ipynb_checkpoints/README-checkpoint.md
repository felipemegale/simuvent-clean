# Gardiner Expy Vehicle Count Analysis

- Maps generated are in ./maps
- Graphs can be found under ./plots

## Jupyter Notebooks
- 000_number_of_readings
    Shows the number of readings and vehicle count on several occasions
    - 000_number_of_readings_per_day
        - Shows how many observations were made throughout the available days
    - 000_number_of_readings_per_month
        - Shows how many observations were made throughout the available months
    - 000_mean_number_of_readings_per_dow
        - Shows the average number of observations per day of week
    - 000_median_number_of_readings_per_dow
        - Shows the median of observation number per day of week
    - 000_total_readings_detector
        - Shows how many observations were made by each detector
    - 000_readings_per_detector_per_dow
        - Shows in log scale the total readings that were made on each day of the week per each detector
    - 000_volume_mean_per_dow
        - Shows the average vehicle count per day of week on a random detector
    - 000_volume_median_per_dow
        - Shows the median vehicle count per day of week on a random detector
    - 000_volume_mean_per_dow_all_locs
        - Shows the average vehicle per day of week of all detectors available
    - 000_volume_median_per_dow_all_locs
        - Shows the median vehicle per day of week of all detectors available
- 010_area_of_interest
    Plots the detector locations on a map
- 020_correlations
    Plots the correlations between numerical variables in the count dataset