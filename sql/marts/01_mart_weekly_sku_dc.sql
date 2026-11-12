-- Grain: week x SKU x DC. The backbone for forecasting, inventory health and service KPIs.
CREATE OR REPLACE TABLE marts.mart_weekly_sku_dc AS
WITH d AS (
    SELECT week_start, sku_id, dc_id, sum(demand_units) AS demand_units, sum(promo_flag::INT) AS promo_days
    FROM staging.stg_demand GROUP BY ALL
), i AS (
    SELECT week_start, sku_id, dc_id,
           sum(sales_units) AS sales_units, sum(lost_sales_units) AS lost_units,
           avg(on_hand_units) AS avg_on_hand, arg_max(on_hand_units, date) AS end_on_hand,
           arg_max(on_order_units, date) AS end_on_order, sum(stockout_flag::INT) AS stockout_days
    FROM staging.stg_inventory GROUP BY ALL
)
SELECT d.week_start, d.sku_id, d.dc_id, s.category, n.region AS dc_region,
       d.demand_units, i.sales_units, i.lost_units, d.promo_days, i.stockout_days,
       i.avg_on_hand, i.end_on_hand, i.end_on_order,
       i.sales_units * s.unit_price AS revenue, i.lost_units * s.unit_price AS lost_revenue,
       i.avg_on_hand * s.unit_cost AS inventory_value,
       CASE WHEN d.demand_units > 0 THEN i.sales_units / d.demand_units END AS fill_rate
FROM d
JOIN i USING (week_start, sku_id, dc_id)
JOIN staging.stg_skus s USING (sku_id)
JOIN staging.stg_nodes n ON n.node_id = d.dc_id;
