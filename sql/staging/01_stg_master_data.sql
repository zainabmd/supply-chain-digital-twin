-- Conformed master data. Views: always reflect the latest raw load.
CREATE OR REPLACE VIEW staging.stg_nodes AS
SELECT node_id, name AS node_name, node_type, tier, region, country, lat, lon,
       capacity_per_week, ttr_days, risk_score, storage_m3
FROM raw.nodes;

CREATE OR REPLACE VIEW staging.stg_edges AS
SELECT src, dst, edge_type, mode, transit_days, cost_per_unit, capacity_per_week, distance_km
FROM raw.edges;

CREATE OR REPLACE VIEW staging.stg_skus AS
SELECT sku_id, category, sku_name, unit_price, unit_cost, unit_volume_m3,
       unit_price - unit_cost AS unit_margin, launch_week
FROM raw.skus;

CREATE OR REPLACE VIEW staging.stg_sourcing AS
SELECT s.sku_id, s.dc_id, s.plant_id, s.planned_lead_time_days, s.learned_lead_time_days,
       n.region AS dc_region
FROM raw.sourcing s
JOIN raw.nodes n ON n.node_id = s.dc_id;
