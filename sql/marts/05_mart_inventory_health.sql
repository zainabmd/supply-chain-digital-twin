-- Grain: SKU x DC, snapshot at the end of history. Starting point for forward-looking risk.
CREATE OR REPLACE TABLE marts.mart_inventory_health AS
WITH last_day AS (SELECT max(date) AS d FROM staging.stg_inventory),
snap AS (
    SELECT i.sku_id, i.dc_id, i.on_hand_units, i.on_order_units
    FROM staging.stg_inventory i, last_day WHERE i.date = last_day.d
), recent AS (
    SELECT sku_id, dc_id, avg(demand_units) AS avg_daily_demand_28d, stddev_samp(demand_units) AS sd_daily_demand_28d
    FROM staging.stg_demand, last_day WHERE date > last_day.d - INTERVAL 28 DAY GROUP BY ALL
), hist AS (
    SELECT sku_id, dc_id, sum(stockout_flag::INT) AS stockout_days_90d,
           sum(sales_units) / NULLIF(sum(sales_units + lost_sales_units), 0) AS fill_rate_90d
    FROM staging.stg_inventory, last_day WHERE date > last_day.d - INTERVAL 90 DAY GROUP BY ALL
), lt AS (
    SELECT sku_id, dc_id, avg(actual_lead_time_days) AS lt_mean_26w, stddev_samp(actual_lead_time_days) AS lt_std_26w
    FROM staging.stg_shipments, last_day WHERE order_date > last_day.d - INTERVAL 182 DAY GROUP BY ALL
)
SELECT snap.*, s.category, s.unit_cost, s.unit_price, s.unit_volume_m3, src.plant_id, src.dc_region,
       src.learned_lead_time_days, coalesce(lt.lt_mean_26w, src.learned_lead_time_days) AS lt_mean_26w,
       coalesce(lt.lt_std_26w, 2.0) AS lt_std_26w,
       recent.avg_daily_demand_28d, recent.sd_daily_demand_28d,
       snap.on_hand_units / NULLIF(recent.avg_daily_demand_28d, 0) AS days_of_cover,
       (snap.on_hand_units + snap.on_order_units) / NULLIF(recent.avg_daily_demand_28d, 0) AS days_of_cover_incl_pipeline,
       snap.on_hand_units * s.unit_cost AS inventory_value,
       hist.stockout_days_90d, hist.fill_rate_90d
FROM snap
JOIN staging.stg_skus s USING (sku_id)
JOIN staging.stg_sourcing src USING (sku_id, dc_id)
LEFT JOIN recent USING (sku_id, dc_id)
LEFT JOIN hist USING (sku_id, dc_id)
LEFT JOIN lt USING (sku_id, dc_id);
