#!/usr/bin/env python
# coding: utf-8
"""
working.py — Supply Chain core logic
Refactored for use as a module imported by app.py (Streamlit).
No global sc_state — everything is passed explicitly or via a cached object.
"""

import os
import ast
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

OUTPUT_DIR = "agent_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# CONFIGURATION
# ==============================
NUM_SKUS     = 500
NUM_STORES   = 10
DAYS_HISTORY = 180
START_DATE   = datetime.today() - timedelta(days=DAYS_HISTORY)
np.random.seed(42)


# ==============================
# SUPPLY CHAIN STATE CLASS
# ==============================
class SupplyChainState:
    def __init__(self, sku_master, central_inventory, demand_history, open_po):
        self.sku_master        = sku_master.copy()
        self.central_inventory = central_inventory.copy()
        self.demand_history    = demand_history.copy()
        self.open_po           = open_po.copy()
        self.action_log        = []
        self.last_updated      = datetime.now()


# ==============================
# LANGGRAPH STATE SCHEMA
# ==============================
class State(TypedDict, total=False):
    sc_state:   object
    messages:   Annotated[list[AnyMessage], add_messages]


# ==============================
# DATA GENERATION
# ==============================
def generate_data() -> SupplyChainState:
    sku_ids     = [f"SKU_{i:04d}" for i in range(1, NUM_SKUS + 1)]
    abc_classes = (
        ["A"] * int(0.2 * NUM_SKUS) +
        ["B"] * int(0.3 * NUM_SKUS) +
        ["C"] * int(0.5 * NUM_SKUS)
    )
    np.random.shuffle(abc_classes)

    sku_master_df = pd.DataFrame({"sku_id": sku_ids, "abc_class": abc_classes})

    def assign_parameters(row):
        if row["abc_class"] == "A":
            return pd.Series({
                "unit_cost":      np.random.uniform(50, 150),
                "selling_price":  np.random.uniform(120, 300),
                "lead_time_days": np.random.randint(7, 15),
                "moq":            np.random.randint(100, 300),
                "safety_stock":   np.random.randint(200, 500),
            })
        elif row["abc_class"] == "B":
            return pd.Series({
                "unit_cost":      np.random.uniform(20, 80),
                "selling_price":  np.random.uniform(80, 150),
                "lead_time_days": np.random.randint(10, 20),
                "moq":            np.random.randint(50, 200),
                "safety_stock":   np.random.randint(100, 300),
            })
        else:
            return pd.Series({
                "unit_cost":      np.random.uniform(5, 30),
                "selling_price":  np.random.uniform(20, 70),
                "lead_time_days": np.random.randint(15, 30),
                "moq":            np.random.randint(20, 100),
                "safety_stock":   np.random.randint(20, 100),
            })

    sku_master_df = pd.concat(
        [sku_master_df, sku_master_df.apply(assign_parameters, axis=1)], axis=1
    )
    sku_master_df["reorder_point"] = sku_master_df["safety_stock"] * 1.2
    sku_master_df["danger_level"]  = sku_master_df["safety_stock"] * 0.5

    # Demand History
    demand_records = []
    for _, row in sku_master_df.iterrows():
        sku, abc = row["sku_id"], row["abc_class"]
        if abc == "A":
            base_demand, variability = np.random.uniform(30, 60), 0.1
        elif abc == "B":
            base_demand, variability = np.random.uniform(10, 30), 0.3
        else:
            base_demand, variability = np.random.uniform(1, 10), 0.6
        for day in range(DAYS_HISTORY):
            date   = START_DATE + timedelta(days=day)
            demand = max(0, np.random.normal(base_demand, base_demand * variability))
            demand_records.append([sku, date, round(demand)])
    demand_history_df = pd.DataFrame(demand_records, columns=["sku_id", "date", "daily_demand"])

    # Central Inventory
    inventory_central_df = sku_master_df[["sku_id"]].copy()
    inventory_central_df["current_stock"]    = np.random.randint(0, 2000, size=NUM_SKUS)
    zero_idx = np.random.choice(NUM_SKUS, 15, replace=False)
    inventory_central_df.loc[zero_idx, "current_stock"] = 0
    inventory_central_df["in_transit_stock"] = np.random.randint(0, 500, size=NUM_SKUS)
    inventory_central_df["last_updated"]     = datetime.today()

    # Open POs
    po_records = [
        [f"PO_{i}", np.random.choice(sku_ids),
         np.random.randint(50, 500),
         datetime.today() + timedelta(days=np.random.randint(5, 25))]
        for i in range(200)
    ]
    open_po_df = pd.DataFrame(
        po_records, columns=["po_id", "sku_id", "ordered_qty", "expected_delivery_date"]
    )

    return SupplyChainState(
        sku_master        = sku_master_df,
        central_inventory = inventory_central_df,
        demand_history    = demand_history_df,
        open_po           = open_po_df,
    )


# ==============================
# DIRECT (NON-LLM) ANALYTICS
# These are called directly by Streamlit — no LLM overhead.
# ==============================

def get_zero_stock_skus(sc: SupplyChainState) -> pd.DataFrame:
    """Return DataFrame of SKUs with zero central stock."""
    zero = sc.central_inventory[sc.central_inventory["current_stock"] <= 0][["sku_id"]].copy()
    zero = zero.merge(sc.sku_master[["sku_id", "abc_class", "unit_cost"]], on="sku_id")
    return zero.sort_values("abc_class").reset_index(drop=True)


def get_below_reorder_skus(sc: SupplyChainState, n: int = 10) -> pd.DataFrame:
    """Return DataFrame of SKUs below reorder point (no open PO), for n-day horizon."""
    demand_df = (
        sc.demand_history
        .groupby("sku_id")
        .agg(total_demand=("daily_demand", "sum"))
        .reset_index()
    )
    demand_df["per_day_consumption"] = (demand_df["total_demand"] / DAYS_HISTORY).round(2)

    merged = sc.central_inventory[["sku_id", "current_stock"]].merge(
        sc.sku_master[["sku_id", "abc_class", "unit_cost"]], on="sku_id"
    ).merge(demand_df[["sku_id", "per_day_consumption"]], on="sku_id")

    merged["reorder_point"] = merged["per_day_consumption"] * n
    below = merged[merged["current_stock"] < merged["reorder_point"]].copy()

    skus_with_po = sc.open_po["sku_id"].unique().tolist()
    no_po = below[~below["sku_id"].isin(skus_with_po)].copy()
    return no_po.sort_values("abc_class").reset_index(drop=True)


def compute_po_recommendations(
    sc: SupplyChainState,
    target_skus: list,
    order_days: int,
    safety_stock_days: int,
) -> pd.DataFrame:
    """
    Given a list of SKU IDs, compute recommended order quantities.
    Returns a DataFrame ready to display and export.
    """
    if not target_skus:
        return pd.DataFrame()

    demand_df = (
        sc.demand_history
        .groupby("sku_id")
        .agg(total_demand=("daily_demand", "sum"))
        .reset_index()
    )
    demand_df["per_day_consumption"] = (demand_df["total_demand"] / DAYS_HISTORY).round(2)

    inventory_df = sc.central_inventory[["sku_id", "current_stock", "in_transit_stock"]]

    po_df = (
        sc.open_po
        .groupby("sku_id")
        .agg(po_quantity=("ordered_qty", "sum"))
        .reset_index()
    )

    df = (
        demand_df
        .merge(inventory_df, on="sku_id")
        .merge(sc.sku_master[["sku_id", "abc_class", "unit_cost", "moq"]], on="sku_id")
        .merge(po_df, on="sku_id", how="left")
    )
    df["po_quantity"]          = df["po_quantity"].fillna(0)
    df = df[df["sku_id"].isin(target_skus)].copy()

    df["total_required_qty"]   = df["per_day_consumption"] * (order_days + safety_stock_days)
    df["available"]            = df["current_stock"] + df["in_transit_stock"] + df["po_quantity"]
    df["final_order_qty"]      = (df["total_required_qty"] - df["available"]).clip(lower=0)
    df["final_order_qty"]      = df.apply(
        lambda r: max(r["final_order_qty"], r["moq"]) if r["final_order_qty"] > 0 else 0, axis=1
    ).round(0)
    df["estimated_value"]      = (df["final_order_qty"] * df["unit_cost"]).round(2)

    df = df[df["final_order_qty"] > 0]

    return df[[
        "sku_id", "abc_class", "per_day_consumption", "current_stock",
        "in_transit_stock", "po_quantity", "total_required_qty",
        "final_order_qty", "moq", "unit_cost", "estimated_value",
    ]].sort_values(["abc_class", "final_order_qty"], ascending=[True, False]).reset_index(drop=True)


def execute_purchase_orders(
    sc: SupplyChainState,
    po_df: pd.DataFrame,
    order_days: int,
    safety_stock_days: int,
    workflow: str,
) -> tuple[pd.DataFrame, str]:
    """
    Appends recommendations to sc.open_po, logs the action,
    saves CSV, and returns (updated_po_df, filepath).
    """
    timestamp = datetime.now()
    new_rows  = []

    for _, row in po_df.iterrows():
        new_rows.append({
            "po_id":                  f"PO_AUTO_{row['sku_id']}_{timestamp.strftime('%Y%m%d%H%M%S')}",
            "sku_id":                 row["sku_id"],
            "ordered_qty":            int(row["final_order_qty"]),
            "expected_delivery_date": timestamp + timedelta(days=14),
        })

    if new_rows:
        sc.open_po = pd.concat(
            [sc.open_po, pd.DataFrame(new_rows)], ignore_index=True
        )

    sc.action_log.append({
        "timestamp":        timestamp.isoformat(),
        "workflow":         workflow,
        "order_days":       order_days,
        "safety_stock_days":safety_stock_days,
        "skus_ordered":     len(po_df),
        "total_value":      round(po_df["estimated_value"].sum(), 2),
    })

    ts_str   = timestamp.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"po_{workflow.lower().replace(' ', '_')}_{ts_str}.csv")
    po_df.to_csv(filepath, index=False)

    return sc.open_po.tail(len(new_rows)).reset_index(drop=True), filepath
