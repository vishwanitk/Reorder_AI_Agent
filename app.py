"""
app.py  —  Supply Chain PO Planner
Run:  streamlit run app.py
Place app.py and working.py in the same folder.
"""

import streamlit as st
import pandas as pd
from working import (
    generate_data,
    get_zero_stock_skus,
    get_below_reorder_skus,
    compute_po_recommendations,
    execute_purchase_orders,
    OUTPUT_DIR,
)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="SC PO Planner",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #f8f9fb; color: #1a1d23; }

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] * { color: #374151 !important; }
section[data-testid="stSidebar"] h3 { color: #111827 !important; font-weight: 600; }

div[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
}
div[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    letter-spacing: .07em;
    text-transform: uppercase;
    font-family: 'DM Mono', monospace !important;
    font-weight: 500;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #111827 !important;
    font-size: 26px !important;
    font-weight: 700;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

div[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}

div.stButton > button {
    background: #1d4ed8;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 14px;
    letter-spacing: .02em;
    padding: 10px 28px;
    transition: all .15s ease;
    width: 100%;
    box-shadow: 0 1px 3px rgba(29,78,216,.3);
}
div.stButton > button:hover {
    background: #1e40af;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(29,78,216,.35);
}

.section-header {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    font-size: 11px;
    letter-spacing: .10em;
    text-transform: uppercase;
    color: #6b7280;
    margin: 28px 0 14px;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 7px;
}

.pill {
    display: inline-block;
    border-radius: 6px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    font-weight: 500;
}
.pill-red    { background:#fef2f2; color:#dc2626; border:1px solid #fecaca; }
.pill-yellow { background:#fffbeb; color:#d97706; border:1px solid #fde68a; }
.pill-blue   { background:#eff6ff; color:#2563eb; border:1px solid #bfdbfe; }

.alert-success {
    background: #f0fdf4; border: 1px solid #bbf7d0;
    border-left: 4px solid #16a34a; border-radius: 8px;
    padding: 14px 18px; color: #15803d;
    font-family: 'DM Mono', monospace; font-size: 13px; margin: 12px 0;
}
.alert-info {
    background: #eff6ff; border: 1px solid #bfdbfe;
    border-left: 4px solid #2563eb; border-radius: 8px;
    padding: 14px 18px; color: #1d4ed8;
    font-family: 'DM Mono', monospace; font-size: 13px; margin: 12px 0;
}
.alert-warn {
    background: #fffbeb; border: 1px solid #fde68a;
    border-left: 4px solid #d97706; border-radius: 8px;
    padding: 14px 18px; color: #92400e;
    font-family: 'DM Mono', monospace; font-size: 13px; margin: 12px 0;
}

.page-title {
    font-family: 'Inter', sans-serif; font-weight: 800;
    font-size: 30px; color: #111827; letter-spacing: -.02em; margin-bottom: 2px;
}
.page-subtitle {
    font-family: 'DM Mono', monospace; font-size: 13px;
    color: #9ca3af; margin-bottom: 28px;
}

.cls-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.cls-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 11px; color: #9ca3af;
    letter-spacing: .08em; text-transform: uppercase;
}
.cls-card-value {
    font-size: 22px; font-weight: 700;
    color: #111827; margin-top: 4px;
}

.log-entry {
    background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 12px 16px; font-family: 'DM Mono', monospace;
    font-size: 12px; color: #6b7280; margin-bottom: 8px;
}
.log-entry span.highlight { color: #1d4ed8; font-weight: 600; }

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #ffffff !important;
    border-color: #d1d5db !important;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────
if "sc" not in st.session_state:
    with st.spinner("Initialising supply chain data…"):
        st.session_state.sc = generate_data()
    st.session_state.po_df         = None
    st.session_state.sku_result    = None
    st.session_state.last_workflow = None
    st.session_state.executed      = False
    st.session_state.exec_result   = None

sc = st.session_state.sc

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    workflow = st.selectbox(
        "Workflow",
        ["Zero Stock", "Below Reorder Point", "Combined"],
        help="Choose which SKUs to include in the PO plan.",
    )

    st.markdown(" ")
    order_days = st.number_input(
        "Order Days", min_value=1, max_value=365, value=30,
        help="Days of demand the order should cover.",
    )
    safety_stock_days = st.number_input(
        "Safety Stock Days", min_value=0, max_value=90, value=5,
        help="Extra buffer days added on top of order days.",
    )

    st.markdown("---")
    run_analysis = st.button("▶  Run Analysis", use_container_width=True)

    if st.button("↺  Reset Session", use_container_width=True):
        for k in ["po_df", "sku_result", "last_workflow", "exec_result"]:
            st.session_state[k] = None
        st.session_state.executed = False
        st.experimental_rerun()

    if sc.action_log:
        st.markdown("---")
        st.markdown("### 📋 Action Log")
        for entry in reversed(sc.action_log[-5:]):
            ts    = entry["timestamp"][:16].replace("T", " ")
            lakhs = entry["total_value"] / 100_000
            st.markdown(
                f'<div class="log-entry">'
                f'<span class="highlight">{entry["workflow"]}</span><br>'
                f'{ts}<br>'
                f'{entry["skus_ordered"]} SKUs · Rs {lakhs:.2f}L'
                f'</div>',
                unsafe_allow_html=True,
            )

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<div class="page-title">📦 Supply Chain PO Planner</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Central warehouse · Purchase order automation</div>',
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total SKUs",        f"{len(sc.sku_master):,}")
c2.metric("Zero Stock SKUs",
          f"{int((sc.central_inventory['current_stock'] <= 0).sum())}",
          delta=f"-{int((sc.central_inventory['current_stock'] <= 0).sum())}",
          delta_color="inverse")
c3.metric("Open POs",          f"{len(sc.open_po)}")
c4.metric("Avg Central Stock", f"{int(sc.central_inventory['current_stock'].mean()):,} u")

# ──────────────────────────────────────────────
# RUN ANALYSIS
# ──────────────────────────────────────────────
if run_analysis:
    st.session_state.executed    = False
    st.session_state.exec_result = None

    with st.spinner("Identifying SKUs…"):
        if workflow == "Zero Stock":
            sku_df      = get_zero_stock_skus(sc)
            target_skus = sku_df["sku_id"].tolist()

        elif workflow == "Below Reorder Point":
            sku_df      = get_below_reorder_skus(sc, n=order_days)
            target_skus = sku_df["sku_id"].tolist()

        else:
            zero_df    = get_zero_stock_skus(sc)
            reorder_df = get_below_reorder_skus(sc, n=order_days)
            sku_df     = (
                pd.concat([zero_df, reorder_df], ignore_index=True)
                .drop_duplicates(subset="sku_id")
                .sort_values("abc_class")
                .reset_index(drop=True)
            )
            target_skus = sku_df["sku_id"].tolist()

        st.session_state.sku_result = {
            "workflow":    workflow,
            "target_skus": target_skus,
            "sku_df":      sku_df,
        }

    with st.spinner("Calculating order quantities…"):
        st.session_state.po_df = compute_po_recommendations(
            sc, target_skus, order_days, safety_stock_days
        )

# ──────────────────────────────────────────────
# RESULTS
# ──────────────────────────────────────────────
if st.session_state.sku_result:
    sr          = st.session_state.sku_result
    po_df       = st.session_state.po_df
    wf_label    = sr["workflow"]
    target_skus = sr["target_skus"]
    sku_df      = sr["sku_df"]

    st.markdown("---")

    # ── Step 1: SKU Identification ──────────────
    st.markdown('<div class="section-header">Step 1 — SKU Identification</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("SKUs Identified",   len(target_skus))
    b.metric("Order Days",        order_days)
    c.metric("Safety Stock Days", safety_stock_days)

    if not sku_df.empty:
        abc_counts = sku_df["abc_class"].value_counts()
        pills = "".join(
            f'<span class="pill pill-{css}">Class {cls}: {abc_counts.get(cls, 0)}</span> &nbsp;'
            for cls, css in [("A", "red"), ("B", "yellow"), ("C", "blue")]
        )
        st.markdown(pills, unsafe_allow_html=True)
        st.markdown(" ")
        display_cols = [
            col for col in
            ["sku_id", "abc_class", "current_stock", "per_day_consumption", "reorder_point"]
            if col in sku_df.columns
        ]
        with st.expander("🔍 View identified SKUs", expanded=False):
            st.dataframe(
                sku_df[display_cols].reset_index(drop=True),
                use_container_width=True,
                height=280,
            )
    else:
        st.markdown(
            '<div class="alert-info">No SKUs matched the selected workflow criteria.</div>',
            unsafe_allow_html=True,
        )

    # ── Step 2: PO Recommendations ──────────────
    st.markdown('<div class="section-header">Step 2 — PO Recommendations</div>', unsafe_allow_html=True)

    if po_df is not None and not po_df.empty:
        lakhs = po_df["estimated_value"].sum() / 100_000
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SKUs Requiring Order", len(po_df))
        m2.metric("Total Order Value",    f"Rs {lakhs:.2f}L")
        m3.metric("Avg Order Qty / SKU",  f"{int(po_df['final_order_qty'].mean()):,}")
        m4.metric("Avg Unit Cost",        f"Rs {po_df['unit_cost'].mean():.0f}")

        abc_val = po_df.groupby("abc_class")["estimated_value"].sum()
        ca, cb, cc = st.columns(3)
        for col, cls in zip([ca, cb, cc], ["A", "B", "C"]):
            v = abc_val.get(cls, 0) / 100_000
            col.markdown(
                f'<div class="cls-card">'
                f'<div class="cls-card-label">Class {cls} Value</div>'
                f'<div class="cls-card-value">Rs {v:.2f}L</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown(" ")

        display_df = po_df.copy()
        display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
        st.dataframe(display_df, use_container_width=True, height=360)

        # ── Step 3: Execute ──────────────────────
        st.markdown(
            '<div class="section-header">Step 3 — Execute Purchase Orders</div>',
            unsafe_allow_html=True,
        )

        if not st.session_state.executed:
            st.markdown(
                f'<div class="alert-warn">⚠️ Review the {len(po_df)} PO recommendations above. '
                f'Confirming will commit Rs {lakhs:.2f}L in orders.</div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"✅  Confirm & Execute {len(po_df)} Purchase Orders",
                use_container_width=True,
            ):
                with st.spinner("Creating purchase orders…"):
                    _, filepath = execute_purchase_orders(
                        sc, po_df, order_days, safety_stock_days, wf_label
                    )
                st.session_state.executed    = True
                st.session_state.exec_result = {
                    "po_count":    len(po_df),
                    "total_value": po_df["estimated_value"].sum(),
                    "filepath":    filepath,
                }
                st.experimental_rerun()

        else:
            res = st.session_state.exec_result
            if res:
                v_l = res["total_value"] / 100_000
                st.markdown(
                    f'<div class="alert-success">'
                    f'✅ <strong>{res["po_count"]} POs executed</strong> · '
                    f'Total value: Rs {v_l:.2f}L · '
                    f'File: <code>{res["filepath"]}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.download_button(
                    label="⬇  Download PO CSV",
                    data=po_df.to_csv(index=False).encode("utf-8"),
                    file_name=res["filepath"].split("/")[-1],
                    mime="text/csv",
                    use_container_width=True,
                )
    else:
        st.markdown(
            '<div class="alert-info">No purchase orders required — '
            'all identified SKUs are sufficiently covered.</div>',
            unsafe_allow_html=True,
        )

else:
    st.markdown("---")
    st.markdown(
        '<div class="alert-info">Select a workflow and parameters in the sidebar, '
        'then click <strong>▶ Run Analysis</strong> to begin.</div>',
        unsafe_allow_html=True,
    )
