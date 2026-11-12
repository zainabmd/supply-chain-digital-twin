-- Grain: week x DC. Executive service KPIs (fill rate, OTIF, days of supply, value).
CREATE OR REPLACE TABLE marts.mart_service_kpis_weekly AS
WITH w AS (
    SELECT week_start, dc_id, any_value(dc_region) AS dc_region,
           sum(demand_units) AS demand_units, sum(sales_units) AS sales_units, sum(lost_units) AS lost_units,
           sum(revenue) AS revenue, sum(lost_revenue) AS lost_revenue, sum(inventory_value) AS inventory_value,
           sum(avg_on_hand) AS avg_on_hand_units
    FROM marts.mart_weekly_sku_dc GROUP BY ALL
), s AS (
    SELECT arrival_week AS week_start, dc_id, count(*) AS shipments,
           avg(on_time::INT) AS otif, avg(actual_lead_time_days) AS avg_lead_time, avg(delay_days) AS avg_delay
    FROM staging.stg_shipments GROUP BY ALL
)
SELECT w.*, s.shipments, s.otif, s.avg_lead_time, s.avg_delay,
       sales_units / NULLIF(demand_units, 0) AS fill_rate,
       avg_on_hand_units / NULLIF(sales_units / 7.0, 0) AS days_of_supply
FROM w LEFT JOIN s USING (week_start, dc_id);
