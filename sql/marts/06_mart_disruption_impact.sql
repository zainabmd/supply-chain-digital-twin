-- Grain: historical disruption event. Measured impact on lead time and lost sales.
CREATE OR REPLACE TABLE marts.mart_disruption_impact AS
WITH ship AS (
    SELECT e.event_id, count(*) AS shipments_affected, avg(s.delay_days) AS avg_delay_days,
           avg(s.on_time::INT) AS on_time_rate
    FROM staging.stg_events e
    JOIN staging.stg_shipments s ON contains(coalesce(s.active_events, ''), e.event_id)
    GROUP BY ALL
), base AS (
    SELECT sum(lost_units) / NULLIF(sum(demand_units), 0) AS baseline_lost_share
    FROM marts.mart_weekly_sku_dc WHERE week_start < DATE '2023-03-01'
), lost AS (
    SELECT e.event_id, sum(w.lost_revenue) AS lost_revenue_during,
           sum(w.lost_units) / NULLIF(sum(w.demand_units), 0) AS lost_share_during
    FROM staging.stg_events e
    JOIN marts.mart_weekly_sku_dc w ON w.week_start BETWEEN e.start_date AND e.end_date + INTERVAL 28 DAY
    GROUP BY ALL
)
SELECT e.*, ship.shipments_affected, ship.avg_delay_days, ship.on_time_rate,
       lost.lost_revenue_during, lost.lost_share_during, base.baseline_lost_share,
       datediff('day', e.start_date, e.end_date) + 1 AS duration_days
FROM staging.stg_events e
LEFT JOIN ship USING (event_id)
LEFT JOIN lost USING (event_id)
CROSS JOIN base;
