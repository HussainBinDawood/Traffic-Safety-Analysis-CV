"""
crossing_verification.py
========================
Two equal panels per row, 4 rows per page (8 panels total):

  LEFT  — Pedestrian space-time diagram (full clip)
           • All other pedestrians: dim, thin, labeled with ID
           • Focus object: full lane colors (R=red, M=orange, L=blue)
           • Calibration zones: waiting (dark blue) / road (dark green)
           • Lane dividers from calibration
           • Waiting time bar shaded on entry side
           • Per-lane time + speed annotations inside lane bands
           • GROUP/INDIV badge, verdict, score, stats block
           • Type flip markers (cyan diamond)

  RIGHT — Vehicle space-time diagram (full clip)
           • Each vehicle a distinct color
           • Crosswalk zone highlighted
           • Blue column = pedestrian wait window (for PET reading)

Usage:
    python crossing_verification.py
    python crossing_verification.py --site site1
    python crossing_verification.py --verdict A
    python crossing_verification.py --clip site1_G_01
"""

from __future__ import annotations
import argparse, math, sys, re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

# ── Paths ─────────────────────────────────────────────────────────────────────
_GDRIVE = (Path.home() / "Library/CloudStorage"
           / "GoogleDrive-a.hariqi@gmail.com/My Drive"
           / "03_Papers-Projects/01_Projects/01_Computer_Vision"
           / "01_Ped_Veh_Detection")
CSV_ROOT   = _GDRIVE / "03_Saved_Data"  / "z_corrected_traj"
CALIB_ROOT = _GDRIVE / "01_applications" / "01_tracking" / "01_calibration"
DIAG_CSV   = (_GDRIVE / "01_applications" / "01_tracking"
              / "03_traj_treatment" / "diagnostic_objects.csv")
OUT_DIR    = _GDRIVE / "01_applications" / "01_tracking" / "03_traj_treatment"

# ── Complete crossing definition ──────────────────────────────────────────────
COMPLETE_SEQS  = {"site1": {"R→M→L","L→M→R"},
                  "site2": {"R→L","L→R"},
                  "site3": {"L→R","R→L"}}
PROG_THRESHOLD = 95.0
_VEH_DIR_OVERRIDES = {"3": "rl"}

# ── Colors — white professional theme ────────────────────────────────────────
LANE_COLORS  = {"R": "#C0392B", "M": "#E67E22", "L": "#2471A3"}
C_INTERP     = "#AAAAAA"
C_OTHER_DIM  = "#BBCCBB"     # light gray-green for other peds
C_OTHER_LBL  = "#888888"     # label color for other peds
VEH_PALETTE  = ["#C0392B","#E67E22","#8E44AD","#1ABC9C",
                "#2471A3","#27AE60","#D35400","#7F8C8D",
                "#16A085","#F39C12","#2980B9","#C0392B"]
C_WAIT_BG    = "#EBF5FB"     # light blue — waiting zone
C_ROAD_BG    = "#EAFAF1"     # light green — road/crossing zone
C_ROAD_EDGE  = "#1E8449"     # dark green road edge
C_LANE_DIV   = "#9B59B6"     # purple lane divider
C_KERB       = "#1A5276"     # dark blue kerb
C_WAIT_BAR   = "#AED6F1"     # blue shade — wait window
C_CX_VEH     = "#D5F5E3"     # light green crosswalk zone
C_BG         = "#FFFFFF"     # white figure background
C_PANEL      = "#FFFFFF"     # white panel background
C_PANEL_VEH  = "#F8F9FA"     # very light gray for vehicle panel
VERDICT_COLS = {"A":"#1E8449","B":"#1A5276","C":"#B7770D","D":"#666666"}

CASES_PER_PAGE = 1   # one case (crossing event) per page
PANEL_W_IN     = 7.0          # each panel width (inches)
PANEL_H_IN     = 3.6          # each panel height (inches)


# ── Calibration ───────────────────────────────────────────────────────────────
_calib_cache: dict = {}
_clip_cache:  dict = {}

def parse_stem(stem: str) -> dict:
    m = re.match(r'^(site\d+)(?:_angle(\d+)_a\d+)?_(I|G)_(\d+)$', stem)
    if m:
        return {"site": m.group(1), "angle": m.group(2) or ""}
    parts = stem.split("_")
    return {"site": next((p for p in parts if p.lower().startswith("site")),
                         "unknown"), "angle": ""}

def load_calib(site: str, angle: str = "") -> dict:
    key = f"{site}_{angle}"
    if key in _calib_cache: return _calib_cache[key]
    site_num = re.search(r'\d+', site).group() if re.search(r'\d+', site) else "1"
    override = _VEH_DIR_OVERRIDES.get(site_num)
    cdir = CALIB_ROOT / f"site_{site_num}"
    cands = ([cdir / f"calibration_a{angle}.txt"] if angle else []) + \
            [cdir / "calibration.txt"]
    cfg = {}
    for cp in cands:
        if cp.exists():
            for line in cp.read_text().splitlines():
                line = line.strip()
                if line.startswith("#") or "=" not in line: continue
                k, _, v = line.partition("=")
                cfg[k.strip()] = v.strip()
            break
    def f(k, d):
        try: return float(cfg.get(k, d))
        except: return float(d)
    def iv(k, d):
        try: return int(float(cfg.get(k, d)))
        except: return int(d)
    veh_dir    = override or cfg.get("veh_direction", "lr")
    n_lanes    = iv("n_lanes", 3)
    road_depth = f("road_depth", 10.15)
    road_y_bot = f("road_y_bot", 2.75);  road_y_top = f("road_y_top", 12.9)
    wait_y_bot = f("wait_y_bot", 0.0);   wait_y_top = f("wait_y_top", 15.65)
    cx_x_left  = f("cx_x_left",  0.0);   cx_x_right = f("cx_x_right", 9.0)
    road_x_left= f("road_x_left", -35.0)
    raw_names  = cfg.get("lane_names", "")
    lane_names = ([s.strip() for s in raw_names.split(",")][:n_lanes]
                  if raw_names else (["R","M","L"] if n_lanes==3 else ["R","L"]))
    lw         = road_depth / n_lanes
    result = {
        "veh_direction": veh_dir, "n_lanes": n_lanes,
        "lane_names": lane_names, "lane_width": lw,
        "road_depth": road_depth,
        "lane_boundaries": [i * lw for i in range(n_lanes + 1)],
        "road_y_bot": road_y_bot, "road_y_top": road_y_top,
        "wait_y_bot": wait_y_bot, "wait_y_top": wait_y_top,
        "cx_x_left": cx_x_left,   "cx_x_right": cx_x_right,
        "road_x_left": road_x_left,
    }
    _calib_cache[key] = result
    return result

def load_clip(clip: str) -> pd.DataFrame | None:
    if clip in _clip_cache: return _clip_cache[clip]
    p = CSV_ROOT / clip / f"{clip}_trajectory_corrected.csv"
    if not p.exists(): _clip_cache[clip] = None; return None
    try:
        df = pd.read_csv(p); _clip_cache[clip] = df; return df
    except Exception:
        _clip_cache[clip] = None; return None


# ── Data loading ──────────────────────────────────────────────────────────────
def load_complete_crossings(site_filter=None, verdict_filter=None,
                            clip_filter=None) -> pd.DataFrame:
    if not DIAG_CSV.exists():
        print(f"[ERROR] {DIAG_CSV} not found"); sys.exit(1)
    diag = pd.read_csv(DIAG_CSV)
    if "flags" in diag.columns:
        diag = diag.rename(columns={"flags": "issue_flags"})
    diag["all_lanes_crossed"] = False
    for site, seqs in COMPLETE_SEQS.items():
        mask = ((diag["site"] == site) &
                (diag["lane_sequence"].isin(seqs)) &
                (diag["prog_max"] >= PROG_THRESHOLD))
        diag.loc[mask, "all_lanes_crossed"] = True
    df = diag[diag["all_lanes_crossed"]].copy()
    if site_filter:    df = df[df["site"].isin(site_filter)]
    if verdict_filter: df = df[df["verdict"].isin(verdict_filter)]
    if clip_filter:    df = df[df["clip"].isin(clip_filter)]
    return df.sort_values(["verdict","site","clip","obj_id"]).reset_index(drop=True)


# ── Timing ────────────────────────────────────────────────────────────────────
def _entry_zone(sub: pd.DataFrame) -> str:
    if "zone" not in sub.columns: return ""
    WAIT = {"waiting_bot","waiting_top","waiting_left","waiting_right"}
    CX   = {"crossing","in_crosswalk","road"}
    cx   = sub[sub["zone"].isin(CX)]
    if cx.empty: return ""
    pre  = sub[(sub["zone"].isin(WAIT)) & (sub["t_s"] < float(cx["t_s"].iloc[0]))]
    return pre["zone"].iloc[-1] if not pre.empty else ""

def compute_timing(sub: pd.DataFrame, calib: dict) -> dict:
    CX   = {"crossing","in_crosswalk","road"}
    cx   = sub[sub["zone"].isin(CX)] if "zone" in sub.columns else pd.DataFrame()
    if "ped_type" in sub.columns:
        lbl = sub["ped_type"].mode().iloc[0]
    elif "ped_class" in sub.columns:
        lbl = "G" if float(sub["ped_class"].mode().iloc[0]) == 2.0 else "P"
    else:
        lbl = sub["obj_type"].mode().iloc[0]
    ez = _entry_zone(sub)
    wait_s = None
    if ez and not cx.empty:
        t0  = float(cx["t_s"].iloc[0])
        pre = sub[(sub["zone"] == ez) & (sub["t_s"] < t0)]
        if not pre.empty:
            wait_s = round(max(0.0, t0 - float(pre["t_s"].iloc[0])), 2)
    # Crossing time = first lane entry → last lane exit
    # (more precise than zone-based; uses actual lane detections)
    t_cross_start = None
    t_cross_end   = None
    if "lane" in sub.columns and not cx.empty:
        cx_lane = cx[cx["lane"].notna() & (cx["lane"].astype(str) != "nan")]
        if not cx_lane.empty:
            t_cross_start = float(cx_lane["t_s"].iloc[0])
            t_cross_end   = float(cx_lane["t_s"].iloc[-1])

    # Fallback to zone-based if no lane data
    if t_cross_start is None and not cx.empty:
        t_cross_start = float(cx["t_s"].iloc[0])
        t_cross_end   = float(cx["t_s"].iloc[-1])

    cross_s = round(t_cross_end - t_cross_start, 2) \
              if (t_cross_start is not None and t_cross_end is not None) else None
    lt = {}; ls_ = {}
    if "lane" in sub.columns and not cx.empty:
        ln_r = cx[cx["lane"].notna() & (cx["lane"].astype(str) != "nan")]
        for ln in calib.get("lane_names", ["R","M","L"]):
            lr = ln_r[ln_r["lane"] == ln]
            if len(lr) >= 2:
                lt[ln]  = round(float(lr["t_s"].iloc[-1]) -
                                float(lr["t_s"].iloc[0]), 2)
                ls_[ln] = round(float(lr["speed_ms"].median()), 2) \
                          if "speed_ms" in lr.columns else 0.0
    return {"ped_label": lbl, "wait_s": wait_s, "cross_s": cross_s,
            "t_cross_start": t_cross_start, "t_cross_end": t_cross_end,
            "entry_zone": ez, "lane_times": lt, "lane_speeds": ls_}


# ── Background zones helper ───────────────────────────────────────────────────
def _draw_ped_bg(ax, calib: dict, t_min: float, t_max: float):
    wy_b = calib["wait_y_bot"]; ry_b = calib["road_y_bot"]
    ry_t = calib["road_y_top"]; wy_t = calib["wait_y_top"]
    lw   = calib["lane_width"]
    # Zone fills
    ax.axhspan(wy_b, ry_b, color=C_WAIT_BG, zorder=0)
    ax.axhspan(ry_b, ry_t, color=C_ROAD_BG, zorder=0)
    ax.axhspan(ry_t, wy_t, color=C_WAIT_BG, zorder=0)
    # Alternating lane tints
    for i in range(calib["n_lanes"]):
        if i % 2 == 0:
            ax.axhspan(ry_b + i*lw, ry_b + (i+1)*lw,
                       color="#ffffff", alpha=0.025, zorder=0)
    # Road edges
    ax.axhline(ry_b, color=C_ROAD_EDGE, lw=1.4, zorder=2)
    ax.axhline(ry_t, color=C_ROAD_EDGE, lw=1.4, zorder=2)
    # Kerb lines
    ax.axhline(wy_b, color=C_KERB, lw=0.8, ls=":", zorder=2)
    ax.axhline(wy_t, color=C_KERB, lw=0.8, ls=":", zorder=2)
    # Lane dividers
    for i in range(1, calib["n_lanes"]):
        ax.axhline(ry_b + i*lw, color=C_LANE_DIV,
                   lw=0.8, ls="--", alpha=0.6, zorder=2)
    # Wait zone labels
    ax.text(t_min + (t_max-t_min)*0.01, (wy_b+ry_b)/2, "WAIT",
            va="center", ha="left", fontsize=5.5,
            color="#446688", alpha=0.9, zorder=3)
    ax.text(t_min + (t_max-t_min)*0.01, (ry_t+wy_t)/2, "WAIT",
            va="center", ha="left", fontsize=5.5,
            color="#446688", alpha=0.9, zorder=3)
    # Lane name labels (right edge)
    for i, ln in enumerate(calib["lane_names"]):
        y_mid = ry_b + (i + 0.5) * lw
        ax.text(t_max - (t_max-t_min)*0.01, y_mid, ln,
                ha="right", va="center", fontsize=7,
                color=LANE_COLORS.get(ln, "#ccc"),
                fontweight="bold", zorder=3)



# ── Legend band helper ────────────────────────────────────────────────────────
def _draw_legend_band(ax, items, bg_color="#F5F5F5"):
    """
    Draw a compact horizontal legend band inside an axes at the very top.
    items: list of (color, linestyle_or_marker, label)
      linestyle: "-", "--", ":", or marker: "o", "s", "D", "rect"
    Returns the fractional height consumed (so the caller can clip the plot).
    """
    ax2 = ax.inset_axes([0.0, 0.935, 1.0, 0.065], zorder=20)
    ax2.set_facecolor(bg_color)
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.axis("off")
    # Separator line at bottom of band
    ax2.axhline(0.04, color="#333355", lw=0.8)

    n      = len(items)
    spacing = 1.0 / n
    pad_x   = spacing * 0.12
    y_icon  = 0.68
    y_text  = 0.06

    for i, (col, style, label) in enumerate(items):
        x_ctr = (i + 0.5) * spacing

        if style == "rect":
            rect = plt.Rectangle(
                (x_ctr - spacing*0.18, 0.30), spacing*0.36, 0.46,
                facecolor=col, alpha=0.45, edgecolor=col, linewidth=0.8,
                transform=ax2.transAxes, clip_on=False)
            ax2.add_patch(rect)
        elif style in ("o","s","D","^"):
            ax2.plot(x_ctr, y_icon, marker=style,
                     color=col, ms=5,
                     markerfacecolor=col,
                     markeredgecolor="#000000", markeredgewidth=0.3,
                     transform=ax2.transAxes, clip_on=False)
        else:
            # Line swatch
            x0 = x_ctr - spacing*0.25
            x1 = x_ctr + spacing*0.25
            lw_ = 2.0 if style == "-" else 1.2
            ax2.plot([x0, x1], [y_icon, y_icon],
                     color=col, lw=lw_, ls=style,
                     transform=ax2.transAxes, clip_on=False)

        ax2.text(x_ctr, y_text, label,
                 ha="center", va="bottom",
                 fontsize=5.6, color="#333333",
                 transform=ax2.transAxes, clip_on=False)

    return ax2


# ── Panel 1: Pedestrian space-time diagram ───────────────────────────────────
def draw_ped_panel(ax, clip_df: pd.DataFrame, focus_id: int,
                   calib: dict, diag_row: pd.Series, t_shared: tuple):
    """
    Pedestrian ST — white background, Y axis = full crossing + waiting geometry.
    Y spans from wait_y_bot to wait_y_top (south kerb to north kerb).
    """
    # ── Style — white professional ────────────────────────────────────────────
    ax.set_facecolor(C_PANEL)
    vc = VERDICT_COLS.get(str(diag_row.get("verdict","?")), "#666")
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(1.0)
    ax.tick_params(colors="#333", labelsize=7, which="both")

    t_min, t_max = t_shared
    t_span = t_max - t_min

    wy_b = calib["wait_y_bot"]   # south kerb = 0.00
    ry_b = calib["road_y_bot"]   # road entry
    ry_t = calib["road_y_top"]   # road exit
    wy_t = calib["wait_y_top"]   # north kerb
    lw_  = calib["lane_width"]
    n_l  = calib["n_lanes"]
    lnames     = calib["lane_names"]
    road_depth = calib["road_depth"]

    # ── Zone fills ────────────────────────────────────────────────────────────
    ax.axhspan(wy_b, ry_b, color=C_WAIT_BG,  zorder=0)   # south wait
    ax.axhspan(ry_b, ry_t, color=C_ROAD_BG,  zorder=0)   # road crossing
    ax.axhspan(ry_t, wy_t, color=C_WAIT_BG,  zorder=0)   # north wait
    # Alternating lane tint
    for i in range(n_l):
        if i % 2 == 0:
            ax.axhspan(ry_b + i*lw_, ry_b + (i+1)*lw_,
                       color="#FFFFFF", alpha=0.6, zorder=0)

    # ── Reference lines ───────────────────────────────────────────────────────
    ax.axhline(wy_b, color=C_KERB,      lw=1.2, ls="--", zorder=3, alpha=0.7)
    ax.axhline(wy_t, color=C_KERB,      lw=1.2, ls="--", zorder=3, alpha=0.7)
    ax.axhline(ry_b, color=C_ROAD_EDGE, lw=1.8, zorder=4)
    ax.axhline(ry_t, color=C_ROAD_EDGE, lw=1.8, zorder=4)
    for i in range(1, n_l):
        ax.axhline(ry_b + i*lw_, color=C_LANE_DIV,
                   lw=0.8, ls="--", alpha=0.5, zorder=3)

    # ── Dimension arrows (left margin) ────────────────────────────────────────
    arr_x = t_min - t_span * 0.055
    def _dim(y0, y1, label, x=None, col="#333333"):
        xp = x or arr_x
        ax.annotate("", xy=(xp, y1), xytext=(xp, y0),
                    arrowprops=dict(arrowstyle="<->", color=col,
                                   lw=0.9, mutation_scale=7),
                    annotation_clip=False, zorder=6)
        ax.text(xp - t_span*0.006, (y0+y1)/2, label,
                ha="right", va="center", fontsize=5.8, color=col,
                rotation=90, clip_on=False, zorder=6)

    _dim(ry_b, ry_t, f"{road_depth:.2f} m  (road crossing)", col="#1E8449")
    x2 = arr_x - t_span * 0.065
    _dim(wy_b, ry_b, f"{ry_b-wy_b:.2f} m  (wait S)", x2, C_KERB)
    _dim(ry_t, wy_t, f"{wy_t-ry_t:.2f} m  (wait N)", x2, C_KERB)

    # Lane width arrows (right margin)
    rx = t_max + t_span * 0.018
    for i, ln in enumerate(lnames):
        y0 = ry_b + i*lw_; y1 = ry_b + (i+1)*lw_
        col = LANE_COLORS.get(ln, "#333")
        ax.annotate("", xy=(rx, y1), xytext=(rx, y0),
                    arrowprops=dict(arrowstyle="<->", color=col,
                                   lw=0.9, mutation_scale=6),
                    annotation_clip=False, zorder=6)
        ax.text(rx + t_span*0.006, (y0+y1)/2, f"{ln}  {lw_:.2f} m",
                ha="left", va="center", fontsize=5.8, color=col,
                clip_on=False, zorder=6)

    # Zone text labels
    ax.text(t_min + t_span*0.01, (wy_b+ry_b)/2, "Waiting (S)",
            va="center", ha="left", fontsize=6, color="#1A5276", zorder=3)
    ax.text(t_min + t_span*0.01, (ry_t+wy_t)/2, "Waiting (N)",
            va="center", ha="left", fontsize=6, color="#1A5276", zorder=3)

    # ── Other pedestrians ─────────────────────────────────────────────────────
    peds = clip_df[clip_df["obj_type"].isin(["P","G"])].copy()
    for oid, grp in peds[peds["obj_id"] != focus_id].groupby("obj_id"):
        grp = grp.sort_values("t_s")
        ax.plot(grp["t_s"].values, grp["y_m"].values,
                color=C_OTHER_DIM, lw=0.9, alpha=0.8, zorder=3)
        mid = len(grp)//2
        ax.text(float(grp["t_s"].iloc[mid]),
                float(grp["y_m"].iloc[mid]),
                f"id {int(oid)}", fontsize=4.5,
                color=C_OTHER_LBL, ha="center", va="bottom",
                alpha=0.8, zorder=4)

    # ── Focus object ──────────────────────────────────────────────────────────
    focus  = peds[peds["obj_id"] == focus_id].sort_values("t_s")
    timing = compute_timing(focus, calib) if not focus.empty else {}

    if not focus.empty:
        has_lane   = "lane"   in focus.columns
        has_interp = "interp" in focus.columns
        has_zone   = "zone"   in focus.columns
        lanes_f    = focus["lane"].values if has_lane \
                     else np.array(["?"]*len(focus))
        interps_f  = focus["interp"].values.astype(bool) if has_interp \
                     else np.zeros(len(focus), bool)
        t_f = focus["t_s"].values
        y_f = focus["y_m"].values

        for i in range(1, len(t_f)):
            is_i = bool(interps_f[i])
            lane = str(lanes_f[i]) \
                   if str(lanes_f[i]) not in ("nan","None","?") else None
            col  = C_INTERP if is_i else LANE_COLORS.get(lane, "#555555")
            lw2  = 0.9 if is_i else 2.5
            ls   = (0,(4,2)) if is_i else "-"
            ax.plot(t_f[i-1:i+1], y_f[i-1:i+1],
                    color=col, lw=lw2, ls=ls,
                    solid_capstyle="round", zorder=6)

        # Road entry / exit markers
        CX = {"crossing","in_crosswalk","road"}
        if has_zone:
            cx_mask = focus["zone"].isin(CX).values
            if cx_mask.any():
                ei = np.where(cx_mask)[0][0]
                xi = np.where(cx_mask)[0][-1]
                ax.plot(t_f[ei], y_f[ei], "o", color="#1E8449", ms=7,
                        markeredgecolor="#FFFFFF", markeredgewidth=1.0,
                        zorder=8)
                ax.plot(t_f[xi], y_f[xi], "s", color="#E67E22", ms=7,
                        markeredgecolor="#FFFFFF", markeredgewidth=1.0,
                        zorder=8)

        # Type flip markers
        flip_str = str(diag_row.get("type_flip_frames",""))
        if flip_str not in ("nan",""):
            for fs in flip_str.split(";"):
                if fs.strip():
                    try:
                        idx = np.where(focus["frame"].values==int(fs))[0]
                        if len(idx):
                            ax.plot(t_f[idx[0]], y_f[idx[0]], "D",
                                    color="#8E44AD", ms=7, zorder=8,
                                    markeredgecolor="#FFFFFF",
                                    markeredgewidth=0.5)
                    except ValueError:
                        pass

        # ── Waiting window bracket ────────────────────────────────────────────
        ez     = timing.get("entry_zone","")
        wait_s = timing.get("wait_s")
        t_road_entry = None
        if ez and wait_s is not None and has_zone:
            cx_check = focus[focus["zone"].isin(CX)]
            if not cx_check.empty:
                t_road_entry = float(cx_check["t_s"].iloc[0])
                pre_w = focus[(focus["zone"]==ez) &
                              (focus["t_s"] < t_road_entry)]
                if not pre_w.empty:
                    t_w0 = float(pre_w["t_s"].iloc[0])
                    t_w1 = float(pre_w["t_s"].iloc[-1])
                    ax.axvspan(t_w0, t_w1, color=C_WAIT_BAR,
                               alpha=0.45, zorder=1)
                    y_brk = wy_b + (ry_b-wy_b)*0.25
                    ax.annotate("", xy=(t_w1, y_brk), xytext=(t_w0, y_brk),
                                arrowprops=dict(arrowstyle="<->",
                                               color="#1A5276",
                                               lw=1.1, mutation_scale=6),
                                zorder=9)
                    ax.text((t_w0+t_w1)/2, y_brk + (ry_b-wy_b)*0.15,
                            f"Wait = {wait_s:.1f} s",
                            ha="center", va="bottom", fontsize=6.5,
                            color="#1A5276", fontweight="bold",
                            bbox=dict(facecolor="#FFFFFF99", pad=1.5,
                                      edgecolor="none"), zorder=10)

        # ── Crossing window band (first lane entry → last lane exit) ───────────
        t_cross_start = timing.get("t_cross_start")
        t_cross_end   = timing.get("t_cross_end")
        cross_s       = timing.get("cross_s")
        if t_cross_start is not None and t_cross_end is not None:
            t_road_exit = t_cross_end
            if t_road_entry is None:
                t_road_entry = t_cross_start
            ax.axvspan(t_cross_start, t_cross_end,
                       color="#1E8449", alpha=0.10, zorder=1)
            ax.axvline(t_cross_start, color="#1E8449",
                       lw=1.4, ls="-", alpha=0.8, zorder=4)
            ax.axvline(t_cross_end,   color="#1E8449",
                       lw=1.4, ls="-", alpha=0.8, zorder=4)
            y_cx_brk = ry_t - (ry_t-ry_b)*0.08
            ax.annotate("", xy=(t_cross_end, y_cx_brk),
                        xytext=(t_cross_start, y_cx_brk),
                        arrowprops=dict(arrowstyle="<->",
                                       color="#1E8449",
                                       lw=1.1, mutation_scale=6),
                        zorder=9)
            ax.text((t_cross_start+t_cross_end)/2,
                    y_cx_brk + (ry_t-ry_b)*0.04,
                    f"Crossing = {cross_s:.1f} s",
                    ha="center", va="bottom", fontsize=6.5,
                    color="#1E8449", fontweight="bold",
                    bbox=dict(facecolor="#FFFFFF99", pad=1.5,
                              edgecolor="none"), zorder=10)

        # ── Per-lane time + speed ─────────────────────────────────────────────
        lt  = timing.get("lane_times",{})
        ls_ = timing.get("lane_speeds",{})
        if has_lane and has_zone:
            cx_rows = focus[focus["zone"].isin(CX)]
            cx_rows = cx_rows[cx_rows["lane"].notna() &
                              (cx_rows["lane"].astype(str)!="nan")]
            for i, ln in enumerate(lnames):
                lr = cx_rows[cx_rows["lane"]==ln]
                if len(lr)>=2 and ln in lt:
                    t0l  = float(lr["t_s"].iloc[0])
                    t1l  = float(lr["t_s"].iloc[-1])
                    t_mid = (t0l+t1l)/2
                    y_mid = ry_b + (i+0.5)*lw_
                    col   = LANE_COLORS.get(ln,"#333")
                    ax.annotate("", xy=(t1l, y_mid+lw_*0.30),
                                xytext=(t0l, y_mid+lw_*0.30),
                                arrowprops=dict(arrowstyle="<->", color=col,
                                               lw=0.9, mutation_scale=5),
                                zorder=8)
                    ax.text(t_mid, y_mid - lw_*0.05,
                            f"{lt[ln]:.1f} s  |  {ls_.get(ln,0):.2f} m/s",
                            ha="center", va="center", fontsize=6.5,
                            color=col, fontweight="bold",
                            bbox=dict(facecolor="#FFFFFFCC", pad=1.5,
                                      edgecolor=col, linewidth=0.7),
                            zorder=9)

    # ── Y axis — full calibrated range ────────────────────────────────────────
    # Y spans: wait_y_bot (south kerb) → wait_y_top (north kerb)
    ax.set_xlim(t_min - t_span*0.09, t_max + t_span*0.09)
    ax.set_ylim(wy_b - 0.2, wy_t + 0.2)
    ax.set_xlabel("Time (s)", fontsize=8, color="#222222", labelpad=4)
    ax.set_ylabel("Y Position (m)", fontsize=8, color="#222222", labelpad=4)

    # Y ticks at every calibration boundary
    y_ticks  = [wy_b, ry_b] + \
               [ry_b + i*lw_ for i in range(1, n_l)] + \
               [ry_t, wy_t]
    y_labels = []
    for yt in y_ticks:
        if   abs(yt - wy_b) < 0.01: y_labels.append(f"{wy_b:.2f}\n(S kerb)")
        elif abs(yt - ry_b) < 0.01: y_labels.append(f"{ry_b:.2f}\n(road in)")
        elif abs(yt - ry_t) < 0.01: y_labels.append(f"{ry_t:.2f}\n(road out)")
        elif abs(yt - wy_t) < 0.01: y_labels.append(f"{wy_t:.2f}\n(N kerb)")
        else:
            # Lane boundary
            idx = round((yt - ry_b) / lw_)
            ln_a = lnames[idx-1] if 0 < idx <= len(lnames) else ""
            ln_b = lnames[idx]   if 0 <= idx < len(lnames) else ""
            y_labels.append(f"{yt:.2f}\n({ln_a}/{ln_b})")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=5.5, color="#333")

    x_ticks = np.arange(math.ceil(t_min/5)*5, t_max+1, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(t)}" for t in x_ticks], fontsize=6.5,
                        color="#333")
    ax.grid(axis="x", color="#E0E0E0", lw=0.6, zorder=1)
    ax.grid(axis="y", color="#E8E8E8", lw=0.4, zorder=1)

    # ── Concise title ─────────────────────────────────────────────────────────
    plbl    = timing.get("ped_label","?")
    wait_s  = timing.get("wait_s")
    cross_s = timing.get("cross_s")
    lseq    = str(diag_row.get("lane_sequence",""))
    w_str   = f"{wait_s:.1f} s"  if wait_s  is not None else "—"
    c_str   = f"{cross_s:.1f} s" if cross_s is not None else "—"
    ax.set_title(
        f"Pedestrian ST  —  {'Group' if plbl=='G' else 'Individual'}   "
        f"Lanes: {lseq}   Wait: {w_str}   Cross: {c_str}",
        fontsize=8, color="#111111", pad=4, fontweight="bold", loc="left")

    # ── Legend band ───────────────────────────────────────────────────────────
    ped_legend = [
        (C_INTERP,   "--",   "Interpolated frames"),
        (C_WAIT_BAR, "rect", "Waiting window"),
        ("#1E8449",  "rect", "Crossing window"),
        ("#1E8449",  "o",    "Road entry"),
        ("#E67E22",  "s",    "Road exit"),
        ("#8E44AD",  "D",    "Type flip"),
    ]
    _draw_legend_band(ax, ped_legend, bg_color="#F5F5F5")


# ── Panel 2: Vehicle space-time diagram ──────────────────────────────────────
def draw_veh_panel(ax, clip_df: pd.DataFrame, calib: dict,
                   focus_df: pd.DataFrame, t_shared: tuple):
    """
    Vehicle ST — white background.
    Y axis = full road extent (road_x_left → cx_x_right) + crosswalk zone.
    """
    ax.set_facecolor(C_PANEL_VEH)
    for sp in ax.spines.values():
        sp.set_edgecolor("#CCCCCC"); sp.set_linewidth(1.0)
    ax.tick_params(colors="#333", labelsize=7)

    t_min, t_max = t_shared
    t_span = t_max - t_min
    cx_l   = calib["cx_x_left"]    # crosswalk left edge
    cx_r   = calib["cx_x_right"]   # crosswalk right edge
    rd_l   = calib["road_x_left"]  # road far left end
    cx_w   = cx_r - cx_l           # crosswalk width
    rd_len = cx_r - rd_l           # total road length

    # Y axis fixed to full road extent from calibration
    y_fixed_bot = rd_l - 0.5     # just below road far end
    y_fixed_top = cx_r + 0.5     # just above crosswalk right edge

    # ── Zone fills ────────────────────────────────────────────────────────────
    ax.axhspan(y_fixed_bot, cx_l, color="#F9F9F9", zorder=0)   # road outside cx
    ax.axhspan(cx_l, cx_r,        color=C_CX_VEH,  zorder=0)   # crosswalk zone
    ax.axhspan(cx_r, y_fixed_top, color="#F9F9F9", zorder=0)   # past crosswalk

    # Reference lines
    ax.axhline(cx_l, color=C_ROAD_EDGE, lw=1.8, zorder=3)
    ax.axhline(cx_r, color=C_ROAD_EDGE, lw=1.8, zorder=3)
    ax.axhline(0.0,  color="#BBBBBB",   lw=0.6, ls=":", zorder=2)

    # Crosswalk label
    ax.text(t_min + t_span*0.01, (cx_l+cx_r)/2,
            "Crosswalk", fontsize=6.5, color="#1E8449",
            va="center", ha="left", zorder=4, style="italic",
            bbox=dict(facecolor="#FFFFFF99", pad=1.5, edgecolor="none"))

    # Traffic direction label
    vdir  = calib["veh_direction"]
    dlbl  = {"lr":"→ L→R","rl":"← R→L","tb":"↓ T→B","bt":"↑ B→T"}.get(vdir,vdir)
    ax.text(t_max - t_span*0.01, y_fixed_top - (y_fixed_top-y_fixed_bot)*0.02,
            f"Traffic: {dlbl}", ha="right", va="top",
            fontsize=6.5, color="#333333",
            bbox=dict(facecolor="#FFFFFF99", pad=1.5, edgecolor="none"),
            zorder=5)

    # ── Dimension arrows (right margin) ───────────────────────────────────────
    rx = t_max + t_span*0.018
    # Crosswalk width
    ax.annotate("", xy=(rx, cx_r), xytext=(rx, cx_l),
                arrowprops=dict(arrowstyle="<->", color="#1E8449",
                               lw=1.0, mutation_scale=7),
                annotation_clip=False, zorder=6)
    ax.text(rx + t_span*0.006, (cx_l+cx_r)/2,
            f"Crosswalk\n{cx_w:.1f} m",
            ha="left", va="center", fontsize=5.8,
            color="#1E8449", clip_on=False, zorder=6)
    # Road length
    rx2 = rx + t_span*0.065
    ax.annotate("", xy=(rx2, cx_r), xytext=(rx2, rd_l),
                arrowprops=dict(arrowstyle="<->", color="#555555",
                               lw=0.9, mutation_scale=7),
                annotation_clip=False, zorder=6)
    ax.text(rx2 + t_span*0.006, (rd_l+cx_r)/2,
            f"Road\n{rd_len:.0f} m",
            ha="left", va="center", fontsize=5.8,
            color="#555555", clip_on=False, zorder=6)

    # ── Pedestrian windows ────────────────────────────────────────────────────
    timing = compute_timing(focus_df, calib) if not focus_df.empty else {}
    ez     = timing.get("entry_zone","")
    wait_s = timing.get("wait_s")
    cross_s= timing.get("cross_s")
    t_road_entry = None

    if ez and not focus_df.empty and "zone" in focus_df.columns:
        CX = {"crossing","in_crosswalk","road"}
        cx_ped = focus_df[focus_df["zone"].isin(CX)]
        if not cx_ped.empty:
            t_road_entry = float(cx_ped["t_s"].iloc[0])
            pre_w = focus_df[(focus_df["zone"]==ez) &
                             (focus_df["t_s"] < t_road_entry)]
            if not pre_w.empty:
                t_w0 = float(pre_w["t_s"].iloc[0])
                t_w1 = float(pre_w["t_s"].iloc[-1])
                # Waiting window
                ax.axvspan(t_w0, t_w1, color=C_WAIT_BAR,
                           alpha=0.45, zorder=1)
                y_w_brk = y_fixed_top - (y_fixed_top-y_fixed_bot)*0.08
                ax.annotate("", xy=(t_w1, y_w_brk), xytext=(t_w0, y_w_brk),
                            arrowprops=dict(arrowstyle="<->", color="#1A5276",
                                           lw=1.1, mutation_scale=6),
                            zorder=9)
                ax.text((t_w0+t_w1)/2,
                        y_w_brk + (y_fixed_top-y_fixed_bot)*0.025,
                        f"Ped wait: {wait_s:.1f} s" if wait_s else "Ped wait",
                        ha="center", va="bottom", fontsize=6.5,
                        color="#1A5276", fontweight="bold",
                        bbox=dict(facecolor="#FFFFFF99", pad=1.5,
                                  edgecolor="none"), zorder=10)

    # Crossing window — first lane entry → last lane exit
    t_cross_start = timing.get("t_cross_start")
    t_cross_end   = timing.get("t_cross_end")
    if t_cross_start is not None and t_cross_end is not None:
        ax.axvspan(t_cross_start, t_cross_end,
                   color="#1E8449", alpha=0.10, zorder=2)
        ax.axvline(t_cross_start, color="#1E8449",
                   lw=1.4, ls="-", alpha=0.8, zorder=4)
        ax.axvline(t_cross_end,   color="#1E8449",
                   lw=1.4, ls="-", alpha=0.8, zorder=4)
        y_cx_brk = y_fixed_top - (y_fixed_top-y_fixed_bot)*0.20
        ax.annotate("", xy=(t_cross_end, y_cx_brk),
                    xytext=(t_cross_start, y_cx_brk),
                    arrowprops=dict(arrowstyle="<->", color="#1E8449",
                                   lw=1.1, mutation_scale=6),
                    zorder=9)
        ax.text((t_cross_start+t_cross_end)/2,
                y_cx_brk + (y_fixed_top-y_fixed_bot)*0.025,
                f"Ped crossing: {cross_s:.1f} s",
                ha="center", va="bottom", fontsize=6.5,
                color="#1E8449", fontweight="bold",
                bbox=dict(facecolor="#FFFFFF99", pad=1.5,
                          edgecolor="none"), zorder=10)

    # Ped enters road (first lane entry)
    if t_cross_start is not None:
        ax.axvline(t_cross_start, color="#1A5276",
                   lw=1.0, ls="--", alpha=0.7, zorder=3)

    # ── Vehicles ──────────────────────────────────────────────────────────────
    vehs = clip_df[clip_df["obj_type"]=="veh"].copy()
    vids = sorted(vehs["obj_id"].unique())
    for vi, vid in enumerate(vids):
        vg  = vehs[vehs["obj_id"]==vid].sort_values("t_s")
        col = VEH_PALETTE[vi % len(VEH_PALETTE)]
        ax.plot(vg["t_s"].values, vg["x_m"].values,
                color=col, lw=1.8, solid_capstyle="round", zorder=5)
        # Label near crosswalk
        cx_vg = vg[(vg["x_m"]>=cx_l-3) & (vg["x_m"]<=cx_r+3)]
        ref   = cx_vg.iloc[0] if not cx_vg.empty else vg.iloc[len(vg)//2]
        ax.text(float(ref["t_s"]) + t_span*0.003,
                float(ref["x_m"]), f" V{int(vid)}",
                fontsize=5.5, color=col,
                ha="left", va="center", zorder=6)

    # ── Y axis — full road + crosswalk from calibration ───────────────────────
    ax.set_xlim(t_min - t_span*0.09, t_max + t_span*0.09)
    ax.set_ylim(y_fixed_bot, y_fixed_top)
    ax.set_xlabel("Time (s)", fontsize=8, color="#222222", labelpad=4)
    ax.set_ylabel("Vehicle X Position (m)", fontsize=8,
                  color="#222222", labelpad=4)

    # Y ticks: road far end, 0, crosswalk edges, and every 5m in between
    step  = 5.0
    auto  = np.arange(math.ceil(rd_l/step)*step, cx_r+step, step)
    fixed = {cx_l, cx_r, 0.0, rd_l}
    y_ticks = sorted(set(float(v) for v in auto) | fixed
                     if y_fixed_bot <= 0 <= y_fixed_top
                     else set(float(v) for v in auto) | {cx_l, cx_r, rd_l})
    y_ticks = [yt for yt in y_ticks if y_fixed_bot <= yt <= y_fixed_top]

    y_lbls = []
    for yt in y_ticks:
        if   abs(yt - rd_l) < 0.5: y_lbls.append(f"{yt:.0f}\n(road end)")
        elif abs(yt - cx_l) < 0.5: y_lbls.append(f"{yt:.0f}\n(cx in)")
        elif abs(yt - cx_r) < 0.5: y_lbls.append(f"{yt:.0f}\n(cx out)")
        elif abs(yt) < 0.5:         y_lbls.append(f"0\n(ref)")
        else:                       y_lbls.append(f"{yt:.0f}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_lbls, fontsize=5.5, color="#333")

    x_ticks = np.arange(math.ceil(t_min/5)*5, t_max+1, 5)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{int(t)}" for t in x_ticks], fontsize=6.5,
                        color="#333")
    ax.grid(axis="x", color="#E0E0E0", lw=0.6, zorder=1)
    ax.grid(axis="y", color="#E8E8E8", lw=0.4, zorder=1)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Vehicle ST  —  {len(vids)} vehicles   "
        f"Crosswalk: {cx_w:.1f} m wide   Road: {rd_len:.0f} m total",
        fontsize=8, color="#111111", pad=4, fontweight="bold", loc="left")

    # ── Legend band ───────────────────────────────────────────────────────────
    veh_legend = [
        (C_WAIT_BAR,  "rect", "Ped waiting window"),
        ("#1E8449",   "rect", "Ped crossing window"),
        ("#1A5276",   "--",   "Ped enters road"),
    ]
    _draw_legend_band(ax, veh_legend, bg_color="#F0F0F0")



# ── Gallery ───────────────────────────────────────────────────────────────────
class VerifyGallery:
    def __init__(self, complete_df: pd.DataFrame):
        self.df       = complete_df
        self.n        = len(complete_df)
        self.n_pages  = self.n   # one case per page
        self.page     = 0
        self.selected = None

        print("[INFO] Loading calibrations...")
        self.calib_map = {}
        for site in complete_df["site"].unique():
            sample = complete_df[complete_df["site"]==site]["clip"].iloc[0]
            meta   = parse_stem(sample)
            self.calib_map[site] = load_calib(site, meta["angle"])
            c = self.calib_map[site]
            print(f"  {site}: {c['n_lanes']} lanes  "
                  f"depth={c['road_depth']}m  veh={c['veh_direction']}")

        print(f"\n[INFO] Loading {complete_df['clip'].nunique()} clip CSVs...")
        miss = sum(1 for cl in complete_df["clip"].unique()
                   if load_clip(cl) is None)
        print(f"[INFO] {complete_df['clip'].nunique()-miss} loaded  "
              f"({miss} not found)\n")

        self._build()

    def _build(self):
        # ── One case per page: 2 equal panels + legend band at bottom ─────────
        # Layout (portrait landscape):
        #   Top bar    : page title + info
        #   Row 0      : [Pedestrian ST | Vehicle ST]  — equal width/height
        #   Legend band: color-coded band spanning full width
        #   Button row : navigation + save

        PANEL_W_IN = 8.5          # each panel width
        PANEL_H_IN = 5.8          # each panel height
        LEG_H_IN   = 0.70         # legend band height
        BTN_H_IN   = 0.38         # button row height
        TOP_H_IN   = 0.55         # title bar height
        GAP_IN     = 0.10         # gap between panels and legend

        fig_w = 2 * PANEL_W_IN + 0.90     # two panels + margins
        fig_h = TOP_H_IN + PANEL_H_IN + GAP_IN + BTN_H_IN + 0.20

        self.fig = plt.figure(figsize=(fig_w, fig_h), facecolor='white')
        self.fig.canvas.manager.set_window_title(
            f"Crossing Verification — {self.n} complete crossings"
        )

        # ── Fractional positions ───────────────────────────────────────────────
        btn_bot   = 0.0
        btn_top   = BTN_H_IN / fig_h
        panel_bot = btn_top + GAP_IN / fig_h
        panel_top = panel_bot + PANEL_H_IN / fig_h

        # ── Two equal panel subplots ──────────────────────────────────────────
        self.gs = gridspec.GridSpec(
            1, 2,
            figure=self.fig,
            top=panel_top,
            bottom=panel_bot,
            left=0.055, right=0.975,
            wspace=0.22
        )
        self.ax_ped = self.fig.add_subplot(self.gs[0, 0])
        self.ax_veh = self.fig.add_subplot(self.gs[0, 1])

        # ── Title + info text ─────────────────────────────────────────────────
        self.title_txt = self.fig.text(
            0.01, 0.995, "", color="#111111",
            fontsize=11, fontweight="bold", va="top")
        self.info_txt = self.fig.text(
            0.01, 0.978, "", color="#aaa", fontsize=8, va="top")

        # Column headers just above panels
        # Column headers removed — titles are inside each panel

        # ── Legend band (full-width, below panels) ────────────────────────────

        # ── Buttons ───────────────────────────────────────────────────────────
        by   = btn_bot + 0.004
        bh_f = BTN_H_IN / fig_h * 0.85
        bs   = dict(color="#EEEEEE", hovercolor="#CCCCCC")

        self.btn_prev = Button(
            self.fig.add_axes([0.01, by, 0.07, bh_f]), "◀ Prev", **bs)
        self.btn_next = Button(
            self.fig.add_axes([0.09, by, 0.07, bh_f]), "Next ▶", **bs)
        self.btn_info = Button(
            self.fig.add_axes([0.22, by, 0.10, bh_f]), "ℹ  Details", **bs)
        self.btn_save = Button(
            self.fig.add_axes([0.78, by, 0.10, bh_f]), "💾  Save PNG",
            color="#D6EAF8", hovercolor="#AED6F1")

        for btn in [self.btn_prev, self.btn_next,
                    self.btn_info, self.btn_save]:
            btn.label.set_color("#111111")
            btn.label.set_fontsize(8.5)

        # Page counter text (centre of button row)
        self.page_txt = self.fig.text(
            0.50, by + bh_f * 0.5, "",
            ha="center", va="center",
            color="#555555", fontsize=8)

        self.btn_prev.on_clicked(self._prev)
        self.btn_next.on_clicked(self._next)
        self.btn_info.on_clicked(self._info)
        self.btn_save.on_clicked(self._save)
        self.fig.canvas.mpl_connect("button_press_event", self._click)
        self.fig.canvas.mpl_connect("key_press_event",    self._key)

        self._render()

    def _render(self):
        self.ax_ped.cla()
        self.ax_veh.cla()

        row     = self.df.iloc[self.page]
        clip    = row["clip"]
        obj_id  = int(row["obj_id"])
        meta    = parse_stem(clip)
        calib   = self.calib_map.get(meta["site"],
                      list(self.calib_map.values())[0])
        clip_df = load_clip(clip)
        focus_df= (clip_df[clip_df["obj_id"]==obj_id]
                   .copy().sort_values("frame")
                   if clip_df is not None else pd.DataFrame())
        t_shared= (float(clip_df["t_s"].min()),
                   float(clip_df["t_s"].max()))                   if clip_df is not None else (0.0, 30.0)

        # Title and info
        short = (clip.replace("site1_","Site 1 — ")
                     .replace("site2_","Site 2 — ")
                     .replace("site3_","Site 3 — "))
        verdict = str(row.get("verdict","?"))
        score   = float(row.get("reliability_score",0))
        vc = VERDICT_COLS.get(verdict, "#aaa")
        self.title_txt.set_text(
            f"Crossing Verification  —  {short}  "
            f"Object {obj_id}  |  Verdict: {verdict}  Score: {score:.0f}/100")
        self.title_txt.set_color(vc)
        self.info_txt.set_text(
            f"Lane sequence: {row.get('lane_sequence','')}   "
            f"Flips: {int(row.get('type_flip_count',0))}   "
            f"Interp: {float(row.get('interp_pct',0)):.0f}%   "
            f"Raw IDs: {int(row.get('n_raw_tracker_ids',0))}   "
            f"Prog: {float(row.get('prog_min',0)):.0f}%→{float(row.get('prog_max',0)):.0f}%"
        )
        self.page_txt.set_text(f"Case {self.page+1} / {self.n}")

        if clip_df is not None:
            draw_ped_panel(self.ax_ped, clip_df, obj_id,
                           calib, row, t_shared)
            draw_veh_panel(self.ax_veh, clip_df, calib,
                           focus_df, t_shared)
        else:
            for ax in (self.ax_ped, self.ax_veh):
                ax.set_facecolor(C_PANEL)
                ax.text(0.5, 0.5, f"CSV not found\n{clip}",
                        transform=ax.transAxes,
                        ha="center", va="center",
                        color="#666", fontsize=10)

        self.fig.canvas.draw_idle()


    def _click(self, event):
        # Single case per page — clicking either panel just prints info
        if event.inaxes in (self.ax_ped, self.ax_veh):
            row = self.df.iloc[self.page]
            print(f"\n  Case {self.page+1}: {row['clip']}  "
                  f"obj={int(row['obj_id'])}  "
                  f"verdict={row['verdict']}  "
                  f"score={row['reliability_score']:.1f}  "
                  f"lanes={row['lane_sequence']}")


    def _prev(self, _):
        if self.page > 0:
            self.page -= 1; self._render()

    def _next(self, _):
        if self.page < self.n_pages - 1:
            self.page += 1; self._render()

    def _info(self, _):
        row      = self.df.iloc[self.page]
        clip_df  = load_clip(row["clip"])
        focus_df = (clip_df[clip_df["obj_id"]==int(row["obj_id"])]
                    .copy().sort_values("frame")
                    if clip_df is not None else pd.DataFrame())
        meta     = parse_stem(row["clip"])
        calib    = self.calib_map.get(meta["site"], {})
        timing   = compute_timing(focus_df, calib) if not focus_df.empty else {}

        print(f"\n  ── Detail ─────────────────────────────────────────")
        for col in ["clip","obj_id","site","verdict","reliability_score",
                    "lane_sequence","direction_mode","prog_max",
                    "type_flip_count","n_raw_tracker_ids","interp_pct",
                    "n_gaps","max_gap_frames","speed_max_crossing","issue_flags"]:
            if col in row.index:
                print(f"    {col:<28}: {row[col]}")
        if timing:
            print(f"\n    label       : {timing.get('ped_label')}")
            print(f"    entry_zone  : {timing.get('entry_zone')}")
            print(f"    wait        : {timing.get('wait_s')} s")
            print(f"    cross       : {timing.get('cross_s')} s")
            for ln, lt in timing.get("lane_times",{}).items():
                print(f"    lane {ln}      : {lt} s  "
                      f"{timing.get('lane_speeds',{}).get(ln,0):.2f} m/s")
        print(f"  ───────────────────────────────────────────────────\n")

    def _save(self, _):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fname = OUT_DIR / f"crossing_verify_{self.page+1:04d}_{self.df.iloc[self.page]['clip']}_obj{int(self.df.iloc[self.page]['obj_id'])}.png"
        self.fig.savefig(str(fname), dpi=150, bbox_inches="tight",
                         facecolor='white')
        print(f"  [SAVED] {fname.name}")

    def _key(self, event):
        k = event.key
        if   k in ("right","n"): self._next(None)
        elif k in ("left", "p"): self._prev(None)
        elif k == "s":           self._save(None)
        elif k == "i":           self._info(None)

    def run(self):
        plt.show(block=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Verify complete crossings: ped ST + vehicle ST side by side."
    )
    parser.add_argument("--site",    nargs="+", default=None)
    parser.add_argument("--verdict", nargs="+", default=None)
    parser.add_argument("--clip",    nargs="+", default=None)
    args = parser.parse_args()

    df = load_complete_crossings(
        site_filter    = args.site,
        verdict_filter = args.verdict,
        clip_filter    = args.clip,
    )
    if df.empty:
        print("[ERROR] No complete crossings found"); sys.exit(1)

    print(f"\n[INFO] {len(df)} complete crossings  ({len(df)} pages, one per case)")
    for site, grp in df.groupby("site"):
        print(f"  {site}: {len(grp)}  "
              f"{grp.verdict.value_counts().to_dict()}")

    print(f"\n  Controls:  ←/→=page  click row=select  I=details  S=save\n")
    VerifyGallery(df).run()


if __name__ == "__main__":
    main()