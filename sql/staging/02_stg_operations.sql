-- Operational facts, typed and enriched with week buckets (ISO weeks, Monday start).
CREATE OR REPLACE VIEW staging.stg_demand AS
SELECT CAST(date AS DATE) AS date, date_trunc('week', CAST(date AS DATE)) AS week_start,
       sku_id, dc_id, demand_units, promo_flag
FROM raw.demand_daily;

CREATE OR REPLACE VIEW staging.stg_inventory AS
SELECT CAST(date AS DATE) AS date, date_trunc('week', CAST(date AS DATE)) AS week_start,
       sku_id, dc_id, on_hand_units, on_order_units, sales_units, lost_sales_units,
       (lost_sales_units > 0) AS stockout_flag
FROM raw.inventory_daily;

CREATE OR REPLACE VIEW staging.stg_shipments AS
SELECT shipment_id, CAST(order_date AS DATE) AS order_date, CAST(arrival_date AS DATE) AS arrival_date,
       date_trunc('week', CAST(arrival_date AS DATE)) AS arrival_week,
       sku_id, dc_id, plant_id, qty_units, planned_lead_time_days, actual_lead_time_days, delay_days, on_time,
       NULLIF(chokepoints, '') AS chokepoints, route_path, NULLIF(active_events, '') AS active_events
FROM raw.shipments;

CREATE OR REPLACE VIEW staging.stg_supplier_deliveries AS
SELECT CAST(week_start AS DATE) AS week_start, supplier_id, plant_id, on_time, days_late, quality_ppm
FROM raw.supplier_deliveries;

CREATE OR REPLACE VIEW staging.stg_events AS
SELECT event_id, name AS event_name, category, kind, nodes, CAST("start" AS DATE) AS start_date,
       CAST("end" AS DATE) AS end_date, delay_days, capacity_factor, description
FROM raw.disruption_events;
