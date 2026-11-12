# Data Dictionary

Units: demand and flows in **units** (finished goods); capacities in **finished-goods units per week**; money in
**USD**; times in **days** unless stated.

## raw (landing files, full refresh)
| Table | Grain | Key columns |
|---|---|---|
| `nodes` | network node | `node_id`, `node_type` (supplier_t3 · supplier_t2 · plant · port · chokepoint · waypoint · dc), `tier`, `region`, `country`, `lat`, `lon`, `capacity_per_week`, `ttr_days` (time to recover), `risk_score` (0–1), `storage_m3` (DCs) |
| `edges` | directed link | `src`, `dst`, `edge_type` (material · component · inland · sea_leg · last_mile), `mode`, `transit_days`, `cost_per_unit`, `capacity_per_week`, `distance_km` |
| `skus` | product | `sku_id` (PHN/LAP/WRB/HOM/PWR-nnn), `category`, `unit_price`, `unit_cost`, `unit_volume_m3`, `base_daily_demand`, `seasonality_amp`, `trend_per_year`, `launch_week` |
| `bom` | SKU × component | `qty_per_unit` |
| `component_sources` / `material_sources` | item × supplier | single-sourced items have one row |
| `sku_plants` | SKU × plant | `role` (primary/secondary), `share` |
| `sourcing` | SKU × DC | `plant_id`, `planned_lead_time_days`, `learned_lead_time_days` (planner estimate at end of history) |
| `demand_daily` | date × SKU × DC | `demand_units` (uncensored orders), `promo_flag` |
| `inventory_daily` | date × SKU × DC | `on_hand_units`, `on_order_units`, `sales_units`, `lost_sales_units` |
| `shipments` | replenishment order | `order_date`, `arrival_date`, `qty_units`, planned and actual lead time, `delay_days`, `on_time` (≤ 3 days late), `chokepoints`, `route_path`, `active_events` |
| `pipeline_open` | in-transit order at end of history | `days_until_arrival`, `qty_units` |
| `promo_calendar` | SKU × region × week | `uplift`; runs 13 weeks past history |
| `disruption_events` | event | `kind` (delay · closure · supplier), `nodes`, `start`, `end`, `delay_days`, `capacity_factor` |
| `supplier_deliveries` | week × tier-2 supplier × plant | `on_time`, `days_late`, `quality_ppm` |

## staging (views)
`stg_nodes`, `stg_edges`, `stg_skus` (+ `unit_margin`), `stg_sourcing` (+ `dc_region`), `stg_demand`,
`stg_inventory` (+ `stockout_flag`), `stg_shipments` (+ `arrival_week`), `stg_supplier_deliveries`, `stg_events`.
Typed, with ISO weeks starting Monday.

## marts
| Table | Grain | Contents |
|---|---|---|
| `mart_weekly_sku_dc` | week × SKU × DC | demand, sales, lost sales, stockout days, on hand, revenue, lost revenue, inventory value, fill rate |
| `mart_service_kpis_weekly` | week × DC | fill rate, OTIF, lead time and delay, days of supply, value |
| `mart_lane_performance` | plant × DC × route | planned, actual and P90 lead time, LT std, on-time rate |
| `mart_chokepoint_exposure` | month × chokepoint | units transiting, avg delay, on-time rate |
| `mart_supplier_scorecard` | supplier | on-time (2 years and H2-24), quality ppm, single-sourced items, TTR, risk |
| `mart_inventory_health` | SKU × DC (snapshot) | on hand, on order, demand and lead-time μ/σ, days of cover, 90-day fill |
| `mart_disruption_impact` | event | shipments affected, avg delay, on-time, lost-sales share during vs baseline |

## ml
| Table | Grain | Contents |
|---|---|---|
| `feature_store_weekly` | origin week × SKU × DC | lags 0/1/2/3/7/12/51, rolling mean/std/max, trend ratio, stockout signal, `feature_version`, `computed_at` |
| `future_covariates` | target week × SKU × DC | promo uplift and flag, week-of-year sin/cos, holiday flags |
| `forecast_weekly` | SKU × DC × horizon 1–13 | `p10`, `p50`, `p90`, `model_version` |
| `forecast_backtest` | SKU × DC × holdout week | predictions, actual `y`, seasonal-naive and MA4 baselines |
| `forecast_feature_importance` | feature | permutation importance of the P50 model |
| `graph_node_metrics` | failable node | `flow_loss_pct`, `revenue_at_risk_weekly` / `_share`, `extra_transit_days`, `ttr_days`, `tts_days`, `uncovered_days`, `risk_exposure_usd`, `betweenness`, `criticality_score`, `is_single_point_of_failure` |
| `graph_min_cut` / `graph_stress_test` | arc / demand multiplier | binding cut; served share vs demand surge |
| `supplier_concentration` | component or material | supplier count, country HHI, top country and region |
| `inventory_policy` | SKU × DC | current vs optimal z, safety stock, reorder point, costs, fill, volume |
| `inventory_policy_dc` / `_summary` / `inventory_frontier` | DC / network / λ | solver status, totals, efficient frontier |
| `stockout_risk_baseline` | SKU × DC × policy | `p_stockout`, `p_any_stockout`, expected fill, lost units and revenue, days to stockout, `risk_band` |
| `control_tower_snapshot` | run | headline KPIs for the dashboard header |

## ops
`dq_results` (check × run), `task_runs` (task × attempt × run), `model_registry` (version × status × metrics JSON).

## Risk definitions
- **Material stockout:** losing at least one full day of expected demand within the 8-week window.
- **Risk band** (by P(material stockout)): Low < 20% ≤ Watch < 50% ≤ High < 80% ≤ Critical.
- **TTS (time to survive):** revenue-weighted days of cover (on hand + pipeline) for the SKU×DCs a node feeds,
  plus a component buffer for upstream suppliers.
- **Risk exposure:** weekly revenue dependent on the node ÷ 7 × max(0, TTR − TTS).
