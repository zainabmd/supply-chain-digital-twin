"""
Master data and topology of the multi-tier global supply network.

    Tier 3  raw-material suppliers  ─┐
    Tier 2  component suppliers     ─┤  bill of materials links the tiers
    Tier 1  assembly plants         ─┘
            origin ports ─▶ sea legs through real chokepoints ─▶ destination ports
            distribution centres (DCs)  ◀── demand

Sea lanes are not abstract plant→DC arrows: every route is a chain of legs
through geographic waypoints (Strait of Malacca, Bab-el-Mandeb, Suez, Cape of
Good Hope, Panama, Hormuz, Taiwan Strait...). Chokepoints are therefore real
graph nodes, so "close Suez" is simply "remove that node". Rerouting, longer
lead times and capacity shortfalls all emerge from the graph, with no
scenario hard-coding.

Every quantity is expressed in **finished-goods-equivalent units per week**,
the standard aggregation for network-level resilience analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import SEED

SEA_KM_PER_DAY = 620.0          # ~17 knots incl. port-call & canal waiting allowance
SEA_COST_PER_UNIT_KM = 0.00042  # USD per unit per km (container cost amortised per unit)

REGIONS = ["Asia Pacific", "Europe", "North America", "Latin America", "Middle East & Africa"]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(min(1.0, a)))


# --------------------------------------------------------------------------- #
# Geography
# --------------------------------------------------------------------------- #
# id: (name, lat, lon, region, country, time-to-recover days if disrupted)
PORTS = {
    "CNSHA": ("Port of Shanghai", 31.23, 121.97, "Asia Pacific", "China", 21),
    "CNYTN": ("Port of Shenzhen (Yantian)", 22.57, 114.28, "Asia Pacific", "China", 18),
    "VNSGN": ("Port of Ho Chi Minh (Cat Lai)", 10.76, 106.79, "Asia Pacific", "Vietnam", 14),
    "SGSIN": ("Port of Singapore", 1.26, 103.84, "Asia Pacific", "Singapore", 10),
    "INNSA": ("Nhava Sheva (Mumbai)", 18.95, 72.95, "Asia Pacific", "India", 14),
    "NLRTM": ("Port of Rotterdam", 51.95, 4.05, "Europe", "Netherlands", 10),
    "DEHAM": ("Port of Hamburg", 53.54, 9.97, "Europe", "Germany", 10),
    "BEANR": ("Port of Antwerp-Bruges", 51.27, 4.33, "Europe", "Belgium", 10),
    "GBFXT": ("Port of Felixstowe", 51.95, 1.33, "Europe", "United Kingdom", 12),
    "USLAX": ("Port of Los Angeles / Long Beach", 33.73, -118.26, "North America", "United States", 21),
    "USNYC": ("Port of New York & New Jersey", 40.67, -74.04, "North America", "United States", 14),
    "USSAV": ("Port of Savannah", 32.08, -81.09, "North America", "United States", 12),
    "BRSSZ": ("Port of Santos", -23.98, -46.30, "Latin America", "Brazil", 16),
    "AEJEA": ("Jebel Ali (Dubai)", 25.01, 55.06, "Middle East & Africa", "UAE", 10),
    "ZADUR": ("Port of Durban", -29.87, 31.03, "Middle East & Africa", "South Africa", 18),
}

# id: (name, lat, lon, capacity share of total weekly demand, is_chokepoint, time-to-recover days)
WAYPOINTS = {
    "CP_TAIWAN": ("Taiwan Strait", 24.3, 119.6, 0.70, True, 60),
    "CP_MALACCA": ("Strait of Malacca", 2.6, 101.2, 1.30, True, 30),
    "CP_BAB_EL_MANDEB": ("Bab-el-Mandeb / Red Sea", 12.6, 43.3, 0.70, True, 120),
    "CP_SUEZ": ("Suez Canal", 30.6, 32.35, 0.70, True, 45),
    "CP_HORMUZ": ("Strait of Hormuz", 26.5, 56.4, 0.25, True, 60),
    "CP_GIBRALTAR": ("Strait of Gibraltar", 35.95, -5.6, 1.00, True, 14),
    "CP_PANAMA": ("Panama Canal", 9.1, -79.7, 0.14, True, 90),
    "CP_CAPE": ("Cape of Good Hope", -35.0, 19.5, 1.50, True, 7),
    "WP_W_AFRICA": ("West Africa sea lane", 12.0, -20.0, 9.99, False, 1),
    "WP_USHANT": ("English Channel approaches", 48.6, -6.5, 9.99, False, 1),
    "WP_WINDWARD": ("Windward Passage", 20.0, -73.9, 9.99, False, 1),
}

# Sea route templates (ordered waypoint chains). Legs are created in both directions.
ORIGIN_PREFIX = {
    "CNSHA": ["CNSHA", "CP_TAIWAN"],
    "CNYTN": ["CNYTN"],
    "VNSGN": ["VNSGN"],
    "SGSIN": ["SGSIN"],
}
EU_PORTS = ["NLRTM", "BEANR", "GBFXT", "DEHAM"]
US_EAST = ["USNYC", "USSAV"]


def _sea_routes() -> list[list[str]]:
    routes: list[list[str]] = []
    for origin, prefix in ORIGIN_PREFIX.items():
        via_malacca = prefix + (["CP_MALACCA"] if origin != "SGSIN" else ["CP_MALACCA"])
        for eu in EU_PORTS:
            routes.append(via_malacca + ["CP_BAB_EL_MANDEB", "CP_SUEZ", "CP_GIBRALTAR", "WP_USHANT", eu])
            routes.append(via_malacca + ["CP_CAPE", "WP_W_AFRICA", "WP_USHANT", eu])
        for us in US_EAST:
            routes.append(via_malacca + ["CP_BAB_EL_MANDEB", "CP_SUEZ", "CP_GIBRALTAR", us])
            routes.append(via_malacca + ["CP_CAPE", us])
            routes.append([origin, "CP_PANAMA", "WP_WINDWARD", us])           # transpacific + canal
        routes.append([origin, "USLAX"])                                       # transpacific west coast
        routes.append(via_malacca + ["CP_CAPE", "BRSSZ"])
        routes.append([origin, "CP_PANAMA", "BRSSZ"])
        routes.append(via_malacca + ["CP_HORMUZ", "AEJEA"])
        routes.append(via_malacca + ["ZADUR"])
        if origin != "SGSIN":
            routes.append(prefix + ["SGSIN"])
    # India (west of Malacca)
    for eu in EU_PORTS:
        routes.append(["INNSA", "CP_BAB_EL_MANDEB", "CP_SUEZ", "CP_GIBRALTAR", "WP_USHANT", eu])
        routes.append(["INNSA", "CP_CAPE", "WP_W_AFRICA", "WP_USHANT", eu])
    for us in US_EAST:
        routes.append(["INNSA", "CP_BAB_EL_MANDEB", "CP_SUEZ", "CP_GIBRALTAR", us])
        routes.append(["INNSA", "CP_CAPE", us])
    routes += [["INNSA", "CP_HORMUZ", "AEJEA"], ["INNSA", "ZADUR"], ["INNSA", "CP_CAPE", "BRSSZ"],
               ["INNSA", "CP_MALACCA", "SGSIN"], ["INNSA", "CP_MALACCA", "USLAX"]]
    # Europe (Hamburg) exports
    for us in US_EAST:
        routes.append(["DEHAM", "WP_USHANT", us])
    routes += [["DEHAM", "WP_USHANT", "BRSSZ"],
               ["DEHAM", "WP_USHANT", "CP_GIBRALTAR", "CP_SUEZ", "CP_BAB_EL_MANDEB", "CP_HORMUZ", "AEJEA"],
               ["DEHAM", "WP_USHANT", "WP_W_AFRICA", "CP_CAPE", "ZADUR"]]
    return routes


PLANTS = {
    # id: (name, lat, lon, region, country, categories, ttr_days, inland legs [(to, days, mode)])
    "PL_SHENZHEN": ("Shenzhen Assembly", 22.65, 114.05, "Asia Pacific", "China",
                    ["Smartphones", "Wearables"], 45, [("CNYTN", 1, "road")]),
    "PL_SUZHOU": ("Suzhou Assembly", 31.30, 120.62, "Asia Pacific", "China",
                  ["Laptops", "Smart Home"], 45, [("CNSHA", 1, "road")]),
    "PL_HCMC": ("Ho Chi Minh Assembly", 10.85, 106.63, "Asia Pacific", "Vietnam",
                ["Smartphones", "Wearables"], 40, [("VNSGN", 1, "road")]),
    "PL_PUNE": ("Pune Assembly", 18.52, 73.86, "Asia Pacific", "India",
                ["Power & Batteries", "Smart Home"], 40, [("INNSA", 2, "road")]),
    "PL_PENANG": ("Penang Assembly", 5.35, 100.28, "Asia Pacific", "Malaysia",
                  ["Laptops", "Wearables"], 35, [("SGSIN", 2, "rail")]),
    "PL_MONTERREY": ("Monterrey Assembly", 25.69, -100.32, "North America", "Mexico",
                     ["Smart Home", "Power & Batteries"], 30, [("USLAX", 3, "rail"), ("USSAV", 4, "rail")]),
    "PL_WROCLAW": ("Wroclaw Assembly", 51.11, 17.03, "Europe", "Poland",
                   ["Power & Batteries", "Smart Home", "Laptops"], 30, [("DEHAM", 2, "road")]),
}

DCS = {
    # id: (name, lat, lon, region, country, size factor, inbound [(from, days, mode)], storage capacity m3)
    "DC_ROTTERDAM": ("Rotterdam DC", 51.92, 4.48, "Europe", "Netherlands", 1.2, [("NLRTM", 1, "road")], 14000),
    "DC_FRANKFURT": ("Frankfurt DC", 50.11, 8.68, "Europe", "Germany", 1.1,
                     [("DEHAM", 2, "rail"), ("NLRTM", 2, "rail"), ("PL_WROCLAW", 2, "road")], 12000),
    "DC_MIDLANDS": ("UK Midlands DC", 52.49, -1.89, "Europe", "United Kingdom", 0.9, [("GBFXT", 1, "road")], 10000),
    "DC_LYON": ("Lyon DC", 45.76, 4.84, "Europe", "France", 0.7,
                [("BEANR", 2, "road"), ("PL_WROCLAW", 3, "road")], 8000),
    "DC_INLAND_EMPIRE": ("Inland Empire DC", 34.06, -117.65, "North America", "United States", 1.5,
                         [("USLAX", 1, "road"), ("PL_MONTERREY", 4, "road")], 18000),
    "DC_DALLAS": ("Dallas DC", 32.78, -96.80, "North America", "United States", 1.0,
                  [("USLAX", 4, "rail"), ("USSAV", 3, "rail"), ("PL_MONTERREY", 2, "road")], 12000),
    "DC_ATLANTA": ("Atlanta DC", 33.75, -84.39, "North America", "United States", 0.9,
                   [("USSAV", 1, "road"), ("PL_MONTERREY", 4, "road")], 10000),
    "DC_NEW_JERSEY": ("New Jersey DC", 40.73, -74.17, "North America", "United States", 1.3,
                      [("USNYC", 1, "road")], 15000),
    "DC_SAO_PAULO": ("São Paulo DC", -23.55, -46.63, "Latin America", "Brazil", 0.8, [("BRSSZ", 1, "road")], 9000),
    "DC_DUBAI": ("Dubai DC", 25.20, 55.27, "Middle East & Africa", "UAE", 0.6, [("AEJEA", 1, "road")], 7000),
    "DC_JOHANNESBURG": ("Johannesburg DC", -26.20, 28.05, "Middle East & Africa", "South Africa", 0.5,
                        [("ZADUR", 2, "rail")], 6000),
    "DC_SINGAPORE": ("Singapore DC", 1.33, 103.75, "Asia Pacific", "Singapore", 0.8,
                     [("SGSIN", 1, "road"), ("PL_PENANG", 2, "road")], 9000),
}

CATEGORIES = {
    # category: (n SKUs, price range USD, components)
    "Smartphones": (14, (199, 1199), ["SoC", "Memory", "Display", "Battery Pack", "Camera Module", "Wireless Module", "Enclosure"]),
    "Laptops": (12, (499, 2499), ["SoC", "Memory", "Display", "Battery Pack", "PCB Assembly", "Enclosure", "Power IC"]),
    "Wearables": (12, (79, 449), ["SoC", "Sensor", "Battery Pack", "Display", "Wireless Module"]),
    "Smart Home": (12, (39, 399), ["SoC", "Wireless Module", "Sensor", "PCB Assembly", "Enclosure", "Power IC"]),
    "Power & Batteries": (10, (25, 299), ["Battery Pack", "Power IC", "PCB Assembly", "Connector", "Enclosure"]),
}

# Explicit SKU prefixes: deriving them from names collided ("Smartphones"/"Smart Home" → "SMA").
CATEGORY_CODES = {"Smartphones": "PHN", "Laptops": "LAP", "Wearables": "WRB", "Smart Home": "HOM", "Power & Batteries": "PWR"}

# component: materials it consumes
COMPONENT_MATERIALS = {
    "SoC": ["Silicon Wafers", "Rare Earths"], "Memory": ["Silicon Wafers"], "Display": ["Glass Substrate", "Rare Earths"],
    "Battery Pack": ["Lithium", "Cobalt"], "Camera Module": ["Glass Substrate", "Silicon Wafers"],
    "Wireless Module": ["Silicon Wafers", "Copper"], "Sensor": ["Silicon Wafers"], "PCB Assembly": ["Copper", "Resin"],
    "Enclosure": ["Aluminium", "Resin"], "Power IC": ["Silicon Wafers", "Copper"], "Connector": ["Copper", "Resin"],
}

# component: [(supplier id, name, country, lat, lon, region, ttr days, risk score 0-1)]
TIER2 = {
    "SoC": [("T2_HSINCHU_FAB", "Hsinchu Foundry", "Taiwan", 24.80, 120.97, "Asia Pacific", 180, 0.55)],  # single source!
    "Memory": [("T2_ICHEON_MEM", "Icheon Memory", "South Korea", 37.27, 127.44, "Asia Pacific", 120, 0.30),
               ("T2_HIROSHIMA_MEM", "Hiroshima Memory", "Japan", 34.39, 132.46, "Asia Pacific", 120, 0.25)],
    "Display": [("T2_PAJU_DISPLAY", "Paju Display", "South Korea", 37.76, 126.78, "Asia Pacific", 90, 0.30),
                ("T2_HEFEI_DISPLAY", "Hefei Display", "China", 31.82, 117.23, "Asia Pacific", 90, 0.40)],
    "Battery Pack": [("T2_NINGDE_CELLS", "Ningde Cells", "China", 26.66, 119.55, "Asia Pacific", 75, 0.35),
                     ("T2_OCHANG_CELLS", "Ochang Cells", "South Korea", 36.71, 127.43, "Asia Pacific", 75, 0.25),
                     ("T2_WROCLAW_CELLS", "Silesia Cells", "Poland", 51.0, 16.9, "Europe", 75, 0.20)],
    "Camera Module": [("T2_SUWON_OPTICS", "Suwon Optics", "South Korea", 37.26, 127.03, "Asia Pacific", 60, 0.25),
                      ("T2_TAICHUNG_OPTICS", "Taichung Optics", "Taiwan", 24.15, 120.67, "Asia Pacific", 60, 0.45)],
    "Wireless Module": [("T2_KAOHSIUNG_RF", "Kaohsiung RF", "Taiwan", 22.63, 120.30, "Asia Pacific", 90, 0.45),
                        ("T2_PENANG_RF", "Penang RF", "Malaysia", 5.35, 100.30, "Asia Pacific", 60, 0.25)],
    "Sensor": [("T2_MUNICH_SENSORS", "Munich Sensors", "Germany", 48.14, 11.58, "Europe", 60, 0.15),
               ("T2_KYOTO_SENSORS", "Kyoto Sensors", "Japan", 35.01, 135.77, "Asia Pacific", 60, 0.20)],
    "PCB Assembly": [("T2_DONGGUAN_PCB", "Dongguan PCB", "China", 23.02, 113.75, "Asia Pacific", 30, 0.35),
                     ("T2_HAIPHONG_PCB", "Haiphong PCB", "Vietnam", 20.84, 106.69, "Asia Pacific", 30, 0.30),
                     ("T2_GUADALAJARA_PCB", "Guadalajara PCB", "Mexico", 20.67, -103.35, "North America", 30, 0.25)],
    "Enclosure": [("T2_DONGGUAN_MECH", "Dongguan Mechanics", "China", 23.05, 113.70, "Asia Pacific", 21, 0.30),
                  ("T2_BANGKOK_MECH", "Bangkok Mechanics", "Thailand", 13.76, 100.50, "Asia Pacific", 21, 0.25)],
    "Power IC": [("T2_HSINCHU_FAB", "Hsinchu Foundry", "Taiwan", 24.80, 120.97, "Asia Pacific", 180, 0.55),
                 ("T2_DRESDEN_FAB", "Dresden Fab", "Germany", 51.05, 13.74, "Europe", 150, 0.15)],
    "Connector": [("T2_SHENZHEN_CONN", "Shenzhen Connectors", "China", 22.55, 114.10, "Asia Pacific", 21, 0.30),
                  ("T2_PUNE_CONN", "Pune Connectors", "India", 18.55, 73.80, "Asia Pacific", 21, 0.30)],
}

# material: [(supplier id, name, country, lat, lon, region, ttr days, risk score)]
TIER3 = {
    "Silicon Wafers": [("T3_NAGANO_WAFER", "Nagano Wafer", "Japan", 36.65, 138.18, "Asia Pacific", 90, 0.20),
                       ("T3_HSINCHU_WAFER", "Hsinchu Wafer", "Taiwan", 24.78, 121.00, "Asia Pacific", 90, 0.50)],
    "Rare Earths": [("T3_BAOTOU_RE", "Baotou Rare Earths", "China", 40.65, 109.84, "Asia Pacific", 150, 0.70)],  # single source
    "Glass Substrate": [("T3_SHIGA_GLASS", "Shiga Glass", "Japan", 35.0, 136.0, "Asia Pacific", 60, 0.20),
                        ("T3_NY_GLASS", "Corning-area Glass", "United States", 42.14, -77.05, "North America", 60, 0.15)],
    "Lithium": [("T3_ATACAMA_LI", "Atacama Lithium", "Chile", -23.65, -68.15, "Latin America", 120, 0.40),
                ("T3_PILBARA_LI", "Pilbara Lithium", "Australia", -20.31, 118.60, "Asia Pacific", 120, 0.25)],
    "Cobalt": [("T3_KATANGA_CO", "Katanga Cobalt", "DR Congo", -10.98, 26.73, "Middle East & Africa", 150, 0.80)],  # single source
    "Copper": [("T3_ESCONDIDA_CU", "Escondida Copper", "Chile", -24.27, -69.07, "Latin America", 60, 0.30),
               ("T3_ZAMBIA_CU", "Copperbelt Copper", "Zambia", -12.80, 28.21, "Middle East & Africa", 60, 0.45)],
    "Resin": [("T3_JUBAIL_RESIN", "Jubail Resin", "Saudi Arabia", 27.01, 49.66, "Middle East & Africa", 30, 0.30),
              ("T3_HOUSTON_RESIN", "Houston Resin", "United States", 29.76, -95.37, "North America", 30, 0.15)],
    "Aluminium": [("T3_QUEBEC_AL", "Quebec Aluminium", "Canada", 48.43, -71.06, "North America", 45, 0.15),
                  ("T3_SHANDONG_AL", "Shandong Aluminium", "China", 36.67, 117.0, "Asia Pacific", 45, 0.35)],
}

DC_SIZES = {k: v[5] for k, v in DCS.items()}


@dataclass
class Network:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    skus: pd.DataFrame
    bom: pd.DataFrame
    sku_plants: pd.DataFrame
    component_sources: pd.DataFrame
    material_sources: pd.DataFrame


def _sku_master(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cat, (n, (lo, hi), _) in CATEGORIES.items():
        for i in range(n):
            price = float(np.round(np.exp(rng.uniform(np.log(lo), np.log(hi))), -0) - 0.01)
            rows.append({
                "sku_id": f"{CATEGORY_CODES[cat]}-{i + 1:03d}", "category": cat,
                "sku_name": f"{cat.rstrip('s')} {chr(65 + i % 26)}{100 + 10 * i}",
                "unit_price": round(price, 2),
                "unit_cost": round(price * rng.uniform(0.48, 0.68), 2),
                "unit_volume_m3": round({"Smartphones": 0.0012, "Laptops": 0.008, "Wearables": 0.0006,
                                         "Smart Home": 0.004, "Power & Batteries": 0.002}[cat] * rng.uniform(0.8, 1.3), 5),
                # Popularity: Pareto-like so a few SKUs dominate volume (as in real portfolios)
                "base_daily_demand": round(float(rng.pareto(1.6) * 6 + 4) * (900 / max(price, 40)) ** 0.35, 2),
                "seasonality_amp": round(rng.uniform(0.15, 0.45), 2),
                "trend_per_year": round(rng.normal(0.08, 0.15), 3),
                # ~15% of the portfolio launches during the history window (new-product ramps).
                "launch_week": int(rng.integers(10, 80)) if rng.random() < 0.15 else 0,
            })
    df = pd.DataFrame(rows)
    if df["sku_id"].duplicated().any():
        raise ValueError("Duplicate SKU ids generated")
    return df


def build_network(seed: int = SEED) -> Network:
    rng = np.random.default_rng(seed)
    skus = _sku_master(rng)

    # ---- SKU -> plants (primary + optional secondary) ----
    sku_plants = []
    for r in skus.itertuples():
        capable = [p for p, v in PLANTS.items() if r.category in v[5]]
        order = rng.permutation(capable)
        sku_plants.append({"sku_id": r.sku_id, "plant_id": order[0], "role": "primary", "share": 1.0})
        if len(order) > 1 and rng.random() < 0.55:
            sku_plants[-1]["share"] = 0.7
            sku_plants.append({"sku_id": r.sku_id, "plant_id": order[1], "role": "secondary", "share": 0.3})
    sku_plants = pd.DataFrame(sku_plants)

    bom = pd.DataFrame([{"sku_id": r.sku_id, "component": c, "qty_per_unit": 1}
                        for r in skus.itertuples() for c in CATEGORIES[r.category][2]])

    # Expected weekly demand (units) per SKU x DC – used to size capacities.
    size_total = sum(DC_SIZES.values())
    weekly_demand_total = float(skus["base_daily_demand"].sum() * 7 * size_total)
    plant_load = {p: 0.0 for p in PLANTS}
    for r in sku_plants.merge(skus[["sku_id", "base_daily_demand"]], on="sku_id").itertuples():
        plant_load[r.plant_id] += r.base_daily_demand * 7 * size_total * r.share

    nodes, edges = [], []

    def add_node(nid, name, ntype, tier, region, country, lat, lon, cap, ttr, risk=0.2):
        nodes.append({"node_id": nid, "name": name, "node_type": ntype, "tier": tier, "region": region,
                      "country": country, "lat": lat, "lon": lon, "capacity_per_week": round(cap, 1),
                      "ttr_days": ttr, "risk_score": risk})

    # Suppliers
    material_sources, component_sources = [], []
    for mat, sups in TIER3.items():
        for sid, name, country, lat, lon, region, ttr, risk in sups:
            material_sources.append({"material": mat, "supplier_id": sid})
            add_node(sid, name, "supplier_t3", 3, region, country, lat, lon,
                     weekly_demand_total * 1.6 / len(sups), ttr, risk)
    seen_t2 = set()
    for comp, sups in TIER2.items():
        for sid, name, country, lat, lon, region, ttr, risk in sups:
            component_sources.append({"component": comp, "supplier_id": sid})
            if sid not in seen_t2:
                seen_t2.add(sid)
                add_node(sid, name, "supplier_t2", 2, region, country, lat, lon,
                         weekly_demand_total * 1.5 / len(sups), ttr, risk)
    material_sources, component_sources = pd.DataFrame(material_sources), pd.DataFrame(component_sources)

    # T3 -> T2 (material supply); T2 -> plant (component supply)
    for comp, sups in TIER2.items():
        for t2 in sups:
            for mat in COMPONENT_MATERIALS[comp]:
                for t3 in TIER3[mat]:
                    km = haversine_km(t3[3], t3[4], t2[3], t2[4])
                    edges.append(_edge(t3[0], t2[0], "material", "sea" if km > 1500 else "road",
                                       transit=max(3, km / SEA_KM_PER_DAY * 1.3 + 4), cost=0.05 + km * 0.00012,
                                       cap=weekly_demand_total * 1.5))
    plant_components = {p: set() for p in PLANTS}
    for r in sku_plants.merge(bom, on="sku_id").itertuples():
        plant_components[r.plant_id].add(r.component)
    for p, comps in plant_components.items():
        plat, plon = PLANTS[p][1], PLANTS[p][2]
        for comp in comps:
            for t2 in TIER2[comp]:
                km = haversine_km(t2[3], t2[4], plat, plon)
                edges.append(_edge(t2[0], p, "component", "air" if km > 6000 and comp in ("SoC", "Memory", "Power IC") else
                                   ("sea" if km > 1500 else "road"),
                                   transit=max(2, km / SEA_KM_PER_DAY * 1.2 + 3) if km > 1500 else 2 + km / 600,
                                   cost=0.1 + km * 0.00015, cap=weekly_demand_total * 1.4))

    # Plants
    for p, (name, lat, lon, region, country, cats, ttr, inland) in PLANTS.items():
        add_node(p, name, "plant", 1, region, country, lat, lon, plant_load[p] * 1.35, ttr, 0.2)
        for to, days, mode in inland:
            edges.append(_edge(p, to, "inland", mode, transit=days, cost=0.6 * days, cap=plant_load[p] * 2))

    # Ports & waypoints
    for pid, (name, lat, lon, region, country, ttr) in PORTS.items():
        add_node(pid, name, "port", 0, region, country, lat, lon, weekly_demand_total * 0.45, ttr, 0.15)
    for wid, (name, lat, lon, share, is_cp, ttr) in WAYPOINTS.items():
        add_node(wid, name, "chokepoint" if is_cp else "waypoint", 0, "Global waters", "—", lat, lon,
                 weekly_demand_total * share, ttr, 0.4 if is_cp else 0.0)

    coords = {n["node_id"]: (n["lat"], n["lon"]) for n in nodes}
    # Sea legs run both ways, EXCEPT out of destination-only ports: otherwise a reroute can "sail through"
    # a port (e.g. Malacca → Durban → West Africa) and silently bypass the Cape of Good Hope chokepoint.
    origin_capable = set(ORIGIN_PREFIX) | {"INNSA", "DEHAM", "USLAX", "USSAV"}
    legs = set()
    for route in _sea_routes():
        for a, b in zip(route, route[1:]):
            for u, v in ((a, b), (b, a)):
                if u in PORTS and u not in origin_capable:
                    continue
                legs.add((u, v))
    for a, b in sorted(legs):
        km = haversine_km(*coords[a], *coords[b])
        edges.append(_edge(a, b, "sea_leg", "sea", transit=km / SEA_KM_PER_DAY, cost=km * SEA_COST_PER_UNIT_KM,
                           cap=weekly_demand_total * 5, km=km))

    # DCs
    for d, (name, lat, lon, region, country, size, inbound, storage) in DCS.items():
        add_node(d, name, "dc", -1, region, country, lat, lon, weekly_demand_total * size / size_total * 1.5, 14, 0.1)
        nodes[-1]["storage_m3"] = storage
        for frm, days, mode in inbound:
            edges.append(_edge(frm, d, "last_mile" if frm in PORTS else "inland", mode, transit=days,
                               cost=0.5 * days + (1.5 if frm in PLANTS else 0.0), cap=weekly_demand_total * size / size_total * 3))

    nodes_df = pd.DataFrame(nodes)
    nodes_df["storage_m3"] = nodes_df.get("storage_m3")
    edges_df = pd.DataFrame(edges).drop_duplicates(subset=["src", "dst"])
    return Network(nodes_df, edges_df.reset_index(drop=True), skus, bom, sku_plants, component_sources, material_sources)


def _edge(src, dst, etype, mode, transit, cost, cap, km=None) -> dict:
    return {"src": src, "dst": dst, "edge_type": etype, "mode": mode, "transit_days": round(float(transit), 2),
            "cost_per_unit": round(float(cost), 4), "capacity_per_week": round(float(cap), 1),
            "distance_km": round(float(km), 1) if km is not None else None}
