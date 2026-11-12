-- Grain: plant x DC route family (chokepoint signature). Lead-time reliability per lane.
CREATE OR REPLACE TABLE marts.mart_lane_performance AS
SELECT plant_id, dc_id, coalesce(chokepoints, 'direct / no chokepoint') AS route_signature,
       count(*) AS shipments, sum(qty_units) AS units,
       avg(planned_lead_time_days) AS planned_lt, avg(actual_lead_time_days) AS actual_lt,
       quantile_cont(actual_lead_time_days, 0.9) AS p90_lt, stddev_samp(actual_lead_time_days) AS lt_std,
       avg(on_time::INT) AS on_time_rate, avg(delay_days) AS avg_delay,
       min(order_date) AS first_used, max(order_date) AS last_used
FROM staging.stg_shipments GROUP BY ALL;

-- Grain: chokepoint x month: how much volume transits each chokepoint, and how late it runs.
CREATE OR REPLACE TABLE marts.mart_chokepoint_exposure AS
WITH x AS (
    SELECT date_trunc('month', order_date) AS month, unnest(string_split(chokepoints, '|')) AS chokepoint,
           qty_units, actual_lead_time_days, delay_days, on_time
    FROM staging.stg_shipments WHERE chokepoints IS NOT NULL
)
SELECT month, chokepoint, count(*) AS shipments, sum(qty_units) AS units,
       avg(actual_lead_time_days) AS avg_lt, avg(delay_days) AS avg_delay, avg(on_time::INT) AS on_time_rate
FROM x GROUP BY ALL;
