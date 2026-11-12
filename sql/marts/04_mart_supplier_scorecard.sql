-- Grain: tier-2/3 supplier. Delivery performance, quality, structural (single-source) risk.
CREATE OR REPLACE TABLE marts.mart_supplier_scorecard AS
WITH perf AS (
    SELECT supplier_id, count(*) AS deliveries, avg(on_time::INT) AS on_time_rate,
           avg(CASE WHEN NOT on_time THEN days_late END) AS avg_days_late, avg(quality_ppm) AS avg_quality_ppm,
           avg(CASE WHEN week_start >= DATE '2024-07-01' THEN on_time::INT END) AS on_time_rate_h2_2024
    FROM staging.stg_supplier_deliveries GROUP BY ALL
), comp_n AS (
    SELECT component, count(*) AS n_sources FROM raw.component_sources GROUP BY ALL
), comps AS (
    SELECT cs.supplier_id, count(*) AS components_supplied,
           sum(CASE WHEN n.n_sources = 1 THEN 1 ELSE 0 END) AS single_sourced_components,
           string_agg(cs.component, ', ') AS components
    FROM raw.component_sources cs JOIN comp_n n USING (component) GROUP BY ALL
), mat_n AS (
    SELECT material, count(*) AS n_sources FROM raw.material_sources GROUP BY ALL
), mats AS (
    SELECT ms.supplier_id, count(*) AS materials_supplied,
           sum(CASE WHEN n.n_sources = 1 THEN 1 ELSE 0 END) AS single_sourced_materials,
           string_agg(ms.material, ', ') AS materials
    FROM raw.material_sources ms JOIN mat_n n USING (material) GROUP BY ALL
)
SELECT n.node_id AS supplier_id, n.node_name AS supplier_name, n.node_type, n.tier, n.region, n.country,
       n.ttr_days, n.risk_score, n.lat, n.lon,
       coalesce(c.components, m.materials) AS supplies,
       coalesce(c.single_sourced_components, 0) + coalesce(m.single_sourced_materials, 0) AS single_source_items,
       p.deliveries, p.on_time_rate, p.avg_days_late, p.avg_quality_ppm, p.on_time_rate_h2_2024
FROM staging.stg_nodes n
LEFT JOIN perf p ON p.supplier_id = n.node_id
LEFT JOIN comps c ON c.supplier_id = n.node_id
LEFT JOIN mats m ON m.supplier_id = n.node_id
WHERE n.node_type IN ('supplier_t2', 'supplier_t3');
