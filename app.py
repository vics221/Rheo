import io
import traceback
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.colors import is_color_like
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests

# =================================================================
# STYLE & COLORS
# =================================================================
C_LIME = "#CFF09E"
C_LEAF = "#A8DBA8"
C_FOREST = "#4D9C73"
C_DEEP = "#245151"
C_OCEAN = "#0B486B"

P_GREEN = ["#002204", "#1B5E20", "#4CAF50", "#81C784", "#C8E6C9", "#F1F8E9"]
P_BLUE  = ["#051937", "#004D72", "#008793", "#4DB6AC", "#B2DFDB", "#E0F2F1"]
P_PINK  = ["#480032", "#880E4F", "#C2185B", "#F06292", "#F8BBD0", "#FFF1F6"]

dropdown_labels = []
for i, c in enumerate(P_GREEN): dropdown_labels.append(f"Green {i+1} ({c})")
for i, c in enumerate(P_BLUE):  dropdown_labels.append(f"Blue {i+1} ({c})")
for i, c in enumerate(P_PINK):  dropdown_labels.append(f"Pink {i+1} ({c})")

ANNOTATION_WIDTH = 1.5
matplotlib.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 18,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10
})

TESTS = ["Frequency", "Amplitude", "Temperature", "Step Shear"]

OUTLIER_FACTOR = 3.0
PLATE_RADIUS_MM = 10.0
GAP_MM = 0.6
MIN_YIELD_STRAIN = 5.0
HIGH_SHEAR_FACTOR = 3.0
END_POINTS = 5


# =================================================================
# EXCEL READING HELPERS (columns are found by header NAME, not position)
# =================================================================
def read_table(df, anchor):
    """Find the header row containing `anchor`; data starts 2 rows below (units row in between)."""
    mask = df.eq(anchor).any(axis=1)
    if not mask.any():
        return None
    r = int(np.flatnonzero(mask.values)[0])
    names, seen = [], {}
    for i, h in enumerate(df.iloc[r]):
        h = str(h).strip() if pd.notna(h) else f"col{i}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        names.append(h)
    body = df.iloc[r + 2:].copy()
    body.columns = names
    return body.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)


def col(d, name):
    low = name.lower()
    for c in d.columns:
        if c.lower() == low:
            return c
    for c in d.columns:
        if c.lower().startswith(low):
            return c
    raise KeyError(f"Column '{name}' not found. Available: {list(d.columns)}")


def classify_sheet(df):
    """Decide which test a sheet belongs to. Uses the sheet title in cell A1 first."""
    title = str(df.iloc[0, 0]).strip().lower() if df.shape[0] and df.shape[1] else ""
    has = lambda s: bool(df.eq(s).any(axis=None))

    if title.startswith("peak hold") or title.startswith("step"):
        return "Step Shear" if (has("Viscosity") and has("Step time")) else None
    if title.startswith("frequency"):
        return "Frequency" if has("Angular frequency") else None
    if title.startswith("amplitude"):
        return "Amplitude" if (has("Oscillation strain") or has("Oscillation displacement")) else None
    if title.startswith("temperature ramp"):
        return "Temperature"

    if has("Angular frequency"):
        try:
            t = read_table(df, "Angular frequency")
            if t[col(t, "Angular frequency")].nunique() > 1:
                return "Frequency"
        except Exception:
            pass
    if has("Oscillation strain"):
        return "Amplitude"
    return None


def scan_files(file_list, progress=None):
    """file_list = [(filename, bytes), ...]  (replaces the old folder scan)"""
    file_list = sorted(file_list, key=lambda x: x[0])
    files = [n for n, _ in file_list]
    info = {"files": files,
            "columns": {t: [] for t in TESTS},
            "sheet_map": {t: {} for t in TESTS},
            "cache": {}}
    for file, data in file_list:
        if progress: progress(f"Læser {file} ...")
        try:
            xl = pd.ExcelFile(io.BytesIO(data))
        except Exception:
            continue
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(xl, sheet_name=sheet, header=None)
                kind = classify_sheet(df)
            except Exception:
                continue
            if kind is None:
                continue
            if kind == "Step Shear":
                info["sheet_map"][kind].setdefault(file, []).append(sheet)
            else:
                info["sheet_map"][kind][file] = sheet
            if file not in info["columns"][kind]:
                info["columns"][kind].append(file)
            info["cache"][(file, sheet)] = df
    info["used_files"] = [f for f in files if any(f in info["columns"][t] for t in TESTS)]
    return info


@st.cache_data(show_spinner="Læser Excel-filer ...")
def scan_uploaded(file_tuple):
    return scan_files(list(file_tuple))


def cached(info, test, file, sheet=None):
    sheet = sheet if sheet is not None else info["sheet_map"][test][file]
    return info["cache"][(file, sheet)]


def read_freq(info, file):
    d = read_table(cached(info, "Frequency", file), "Angular frequency")
    cw, c1, c2, ct = col(d, "Angular frequency"), col(d, "Storage modulus"), col(d, "Loss modulus"), col(d, "Tan(delta)")
    d = d.dropna(subset=[cw, c1, c2, ct])
    return dict(x=d[cw].values, g1=d[c1].values, g2=d[c2].values, tan=d[ct].values)


def read_amp(info, file):
    d = read_table(cached(info, "Amplitude", file), "Storage modulus")
    c1, c2 = col(d, "Storage modulus"), col(d, "Loss modulus")
    derived = False
    try:
        cx = col(d, "Oscillation strain")
        x = d[cx]
    except KeyError:
        cx = col(d, "Oscillation displacement")
        x = d[cx] * 100.0 * PLATE_RADIUS_MM / GAP_MM
        derived = True
    d = pd.DataFrame({"x": x, "g1": d[c1], "g2": d[c2]}).dropna()
    return dict(x=d["x"].values, g1=d["g1"].values, g2=d["g2"].values, derived=derived)


def read_temp(info, file):
    df = cached(info, "Temperature", file)
    d = read_table(df, "Storage modulus")
    try:
        cx, c1, c2 = col(d, "Temperature"), col(d, "Storage modulus"), col(d, "Loss modulus")
        d = d.dropna(subset=[cx, c1, c2])
        return dict(x=d[cx].values, g1=d[c1].values, g2=d[c2].values)
    except Exception:
        d = df.apply(pd.to_numeric, errors="coerce").dropna(subset=[0, 1, 6])
        return dict(x=d[6].values, g1=d[0].values, g2=d[1].values)


def read_steps(info, file):
    """Return the list of Peak hold steps (in order) for one file."""
    steps = []
    for sheet in info["sheet_map"]["Step Shear"][file]:
        d = read_table(cached(info, "Step Shear", file, sheet), "Step time")
        if d is None:
            continue
        ct, cv = col(d, "Step time"), col(d, "Viscosity")
        try:
            cs = col(d, "Shear rate")
        except KeyError:
            cs = None
        d = d.dropna(subset=[ct, cv])
        if d.empty:
            continue
        t, v = d[ct].values, d[cv].values
        med = np.median(v)
        keep = (v <= med * OUTLIER_FACTOR) & (v >= med / OUTLIER_FACTOR) if med > 0 else np.ones(len(v), bool)
        if not keep.any():
            continue
        steps.append(dict(
            t=t[keep], v=v[keep], duration=float(t[-1]),
            shear=float(np.nanmedian(d[cs].values)) if cs else np.nan,
            end=float(np.mean(v[keep][-END_POINTS:])),
        ))
    shears = np.array([s["shear"] for s in steps])
    if len(steps) and np.all(np.isfinite(shears)):
        lo = shears.min()
        for s in steps:
            s["high"] = bool(s["shear"] > HIGH_SHEAR_FACTOR * lo)
    else:
        for i, s in enumerate(steps):
            s["high"] = (i % 2 == 1)
    return steps


def recoveries(steps):
    """Recovery % = viscosity at end of each later low-shear step / end of the FIRST low-shear step."""
    low = [s for s in steps if not s["high"]]
    if len(low) < 2 or low[0]["end"] <= 0:
        return []
    return [100.0 * s["end"] / low[0]["end"] for s in low[1:]]


def aggregate(x_list, y_lists, log_x):
    """Pool replicates, bin on x, return binned mean x and (mean, SEM) for every y."""
    df_c = pd.DataFrame({'x': np.concatenate(x_list),
                         **{k: np.concatenate(v) for k, v in y_lists.items()}})
    if log_x:
        df_c = df_c[df_c['x'] > 0].copy()
        df_c['k'] = np.log10(df_c['x']).round(2)
    else:
        df_c['k'] = df_c['x'].round(1)
    agg = df_c.groupby('k').agg({'x': ['mean'], **{k: ['mean', 'sem'] for k in y_lists}})
    x_m = agg['x']['mean'].values
    return x_m, {k: (agg[k]['mean'].values, agg[k]['sem'].fillna(0).values) for k in y_lists}


def yield_crossover(x, g1, g2, min_x=MIN_YIELD_STRAIN):
    """Strain where G' and G'' cross (linear interpolation), only accepting crossings at strain >= min_x."""
    diff, avg = g1 - g2, (g1 + g2) / 2
    for i in range(len(diff) - 1):
        if diff[i] == 0 and x[i] >= min_x:
            return x[i], avg[i]
        if diff[i] * diff[i + 1] < 0:
            x1, x2, d1, d2 = x[i], x[i + 1], diff[i], diff[i + 1]
            ys = x1 + (0 - d1) * (x2 - x1) / (d2 - d1)
            if ys >= min_x:
                yv = avg[i] + (ys - x1) * (avg[i + 1] - avg[i]) / (x2 - x1)
                return ys, yv
    return None


def mean_sem(vals):
    """Mean and standard error of the mean (sample SD / sqrt(n)). SEM is 0 when n = 1."""
    v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return float(v[0]), 0.0
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


# =================================================================
# PER-TEST PLOTTING (one sample group on one axis)
# =================================================================
def plot_frequency(ax, info, files, label, color, store):
    reps = [read_freq(info, f) for f in files]

    idx10 = [int(np.abs(r["x"] - 10.0).argmin()) for r in reps]
    v_g1 = [r["g1"][i] for r, i in zip(reps, idx10)]
    v_g2 = [r["g2"][i] for r, i in zip(reps, idx10)]
    v_tn = [r["tan"][i] for r, i in zip(reps, idx10)]
    store["rep"]["modulus"].setdefault(label, []).extend(v_g1)
    store["rep"]["tan"].setdefault(label, []).extend(v_tn)
    (m1, e1), (m2, e2) = mean_sem(v_g1), mean_sem(v_g2)
    store["mod_bars"].append(dict(label=label, color=color, g1=m1, g2=m2, g1e=e1, g2e=e2))

    x_m, a = aggregate([r["x"] for r in reps],
                       {"g1": [r["g1"] for r in reps], "g2": [r["g2"] for r in reps], "tan": [r["tan"] for r in reps]},
                       log_x=True)
    (g1, g1s), (g2, g2s), (tn, tns) = a["g1"], a["g2"], a["tan"]

    ax.errorbar(x_m, g1, yerr=g1s, marker='o', ls='None', color=color, ms=8, capsize=3, label="G'")
    ax.errorbar(x_m, g2, yerr=g2s, marker='o', ls='None', color=color, mfc='None', ms=8, capsize=3, label="G''")

    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1, 10000)
    ax.set_xlabel('Angular frequency (rad/s)'); ax.set_ylabel('Pascal')

    ax2 = ax.twinx()
    ax2.errorbar(x_m, tn, yerr=tns, marker='^', ls='', color=C_DEEP, ms=8, alpha=0.5, capsize=3, label="tan(δ)")
    ax2.set_ylabel('tan(δ)'); ax2.set_ylim(0, 1.5)
    x_min = np.min(x_m)
    ax2.axhline(1.0, color=C_DEEP, linestyle='--', linewidth=ANNOTATION_WIDTH)
    ax2.axhspan(0, 1.0, facecolor=C_FOREST, alpha=0.08)
    ax2.text(x_min * 0.85, 0.93, '   SOLID', fontweight='bold', color=C_FOREST, va='center')
    ax2.text(x_min * 0.85, 1.07, '   LIQUID', fontweight='bold', color=C_OCEAN, va='center')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper left', frameon=True)
    store["processed"]["Frequency"][label] = dict(x=x_m, g1=g1, g2=g2, tan=tn, g1s=g1s, g2s=g2s, tans=tns, color=color)


def plot_amplitude(ax, info, files, label, color, store):
    reps = [read_amp(info, f) for f in files]
    derived = any(r["derived"] for r in reps)
    if derived:
        store["notes"].add(f"Amplitude sweep of '{label}' had no strain column: strain was calculated from "
                           f"oscillation displacement (plate radius {PLATE_RADIUS_MM:g} mm, gap {GAP_MM:g} mm).")

    for k, r in enumerate(reps):
        ax.plot(r["x"], r["g1"], marker='o', ls='', color=color, ms=6, alpha=0.8, label="G'" if k == 0 else None)
        ax.plot(r["x"], r["g2"], marker='o', ls='', color=color, mfc='None', ms=6, alpha=0.8, label="G''" if k == 0 else None)

    results = [yield_crossover(r["x"], r["g1"], r["g2"]) for r in reps]
    found = [res for res in results if res is not None]
    ys_vals = [res[0] for res in found]
    if ys_vals:
        store["rep"]["yield"].setdefault(label, []).extend(ys_vals)

    x_min = min(r["x"].min() for r in reps)
    if x_min < MIN_YIELD_STRAIN:
        ax.axvspan(x_min, MIN_YIELD_STRAIN, color='grey', alpha=0.10, lw=0)

    bar = dict(label=label, color=color, val=np.nan, err=np.nan, note=f"No crossover\nabove {MIN_YIELD_STRAIN:g}%")
    if found:
        ys, ys_sem = mean_sem(ys_vals)
        yv = float(np.mean([res[1] for res in found]))
        bar.update(val=ys, err=ys_sem)
        txt = f'{ys:.3g}' + (f'±{ys_sem:.2g}' if len(found) > 1 else '') + '%'
        if len(found) < len(reps):
            txt += f'\n(n={len(found)} of {len(reps)})'
        ax.axvline(x=ys, color=C_DEEP, ls='--', lw=ANNOTATION_WIDTH, alpha=0.8)
        ax.plot(ys, yv, marker='*', color=C_LIME, ms=18, mec='k', zorder=10)
        ax.annotate(txt, xy=(ys, yv), xytext=(-14, 26), textcoords='offset points', ha='right', va='bottom',
                    color=C_DEEP, fontweight='bold', zorder=11,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=0.85),
                    arrowprops=dict(arrowstyle='-', color=C_DEEP, lw=1))
    elif all(np.all(r["g2"] > r["g1"]) for r in reps):
        bar["note"] = "Material is\nnot a gel"
        ax.text(0.5, 0.9, "Material is not a gel", color='red', fontweight='bold', fontsize=13,
                ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round,pad=0.3'))
    else:
        ax.text(0.5, 0.9, f"No crossover above {MIN_YIELD_STRAIN:g}%", color='darkorange', fontweight='bold',
                fontsize=13, ha='center', va='center', transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='darkorange', boxstyle='round,pad=0.3'))
    store["yield_bars"].append(bar)

    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(1, 10000)
    ax.set_xlabel('Strain (%, calculated)' if derived else 'Strain (%)'); ax.set_ylabel('Pascal')
    ax.legend(loc='upper left')
    store["processed"]["Amplitude"][label] = dict(reps=reps, color=color)


def plot_temperature(ax, info, files, label, color, store):
    reps = [read_temp(info, f) for f in files]
    if sum(len(r["x"]) for r in reps) == 0:
        raise ValueError("No temperature data found")
    for k, r in enumerate(reps):
        ax.plot(r["x"], r["g1"], marker='o', ms=4, ls='-', lw=0.8, color=color, alpha=0.8, label="G'" if k == 0 else None)
        ax.plot(r["x"], r["g2"], marker='o', ms=4, ls='-', lw=0.8, color=color, mfc='None', alpha=0.8, label="G''" if k == 0 else None)
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Pascal"); ax.set_yscale("log")
    ax.legend()
    store["processed"]["Temperature"][label] = dict(reps=reps, color=color)


def plot_step_shear(ax, info, files, label, color, store):
    reps = [read_steps(info, f) for f in files]
    reps = [r for r in reps if r]
    if not reps:
        raise ValueError("No Peak hold steps with viscosity data")

    rec_all = [recoveries(r) for r in reps]
    n_cyc = max((len(r) for r in rec_all), default=0)
    cyc_mean, cyc_sem = [], []
    for k in range(n_cyc):
        vals = [r[k] for r in rec_all if len(r) > k]
        cyc_mean.append(np.mean(vals))
        cyc_sem.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        if k == 0:
            store["rep"]["recovery"][label] = list(vals)
    if n_cyc:
        store["rec_bars"].append(dict(label=label, color=color, mean=cyc_mean, sem=cyc_sem))

    T, V, S, bounds = [], [], [], []
    offset = 0.0
    for i in range(max(len(r) for r in reps)):
        present = [r[i] for r in reps if len(r) > i]
        dur = max(p["duration"] for p in present)
        if len(present) == 1:
            t_rel, mean_v, sem_v = present[0]["t"], present[0]["v"], np.zeros(len(present[0]["t"]))
        else:
            n_pts = max(len(p["t"]) for p in present)
            t_rel = np.linspace(min(p["t"][0] for p in present), dur, n_pts)
            arr = np.array([np.interp(t_rel, p["t"], p["v"]) for p in present])
            mean_v = arr.mean(axis=0)
            sem_v = arr.std(axis=0, ddof=1) / np.sqrt(len(present))
        T.append(offset + t_rel); V.append(mean_v); S.append(sem_v)
        bounds.append(dict(start=offset, end=offset + dur, high=present[0]["high"],
                           level=float(np.mean([p["end"] for p in present]))))
        offset += dur

    x_m, y_m, y_s = np.concatenate(T), np.concatenate(V), np.concatenate(S)

    for b in bounds:
        if b["high"]:
            ax.axvspan(b["start"], b["end"], color='grey', alpha=0.15, lw=0)
    ax.plot(x_m, y_m, marker='o', ls='', color=color, ms=4, alpha=0.7)
    if np.any(y_s > 0):
        ax.fill_between(x_m, y_m - y_s, y_m + y_s, color=color, alpha=0.25, lw=0)

    low = [b for b in bounds if not b["high"]]
    if len(low) >= 2:
        ax.axhline(low[0]["level"], color=C_DEEP, ls='--', lw=ANNOTATION_WIDTH, alpha=0.5)
        for k, b in enumerate(low[1:]):
            if k < n_cyc:
                txt = f'{cyc_mean[k]:.0f}%' + (f'±{cyc_sem[k]:.0f}' if cyc_sem[k] >= 0.5 else '')
                ax.annotate(txt, xy=(b["end"], b["level"]), xytext=(0, 14), textcoords='offset points',
                            ha='center', color=C_DEEP, fontweight='bold',
                            arrowprops=dict(arrowstyle='-', color=C_DEEP, lw=1))

    ax.set_xlabel('Total time (s)'); ax.set_ylabel('Viscosity (Pa·s)')
    ax.margins(y=0.15)
    ax.set_ylim(bottom=0)
    store["processed"]["Step Shear"][label] = dict(x=x_m, y=y_m, s=y_s, color=color)


PLOTTERS = {"Frequency": plot_frequency, "Amplitude": plot_amplitude,
            "Temperature": plot_temperature, "Step Shear": plot_step_shear}


# =================================================================
# STATISTICS
#   1. per group: n, mean, SEM
#   2. one-way ANOVA across the groups
#   3. only if ANOVA p < 0.05: pairwise Welch t-tests, Benjamini-Hochberg adjusted
# =================================================================
def run_stats(title, data):
    excluded = [k for k, v in data.items() if len(v) > 0 and "polyonly" in k.lower()]
    data = {k: list(v) for k, v in data.items() if len(v) > 0 and "polyonly" not in k.lower()}
    lines = [title]
    for k, v in data.items():
        m, e = mean_sem(v)
        lines.append(f"  {k}: n = {len(v)}, mean = {m:.4g}" + (f", SEM = {e:.3g}" if len(v) > 1 else ", SEM needs n >= 2"))
    if excluded:
        lines.append(f"  (excluded from statistics: {', '.join(excluded)})")

    names = list(data)
    if len(names) < 2:
        lines.append("  ANOVA not run: it needs at least 2 groups.")
        return "n/a", "\n".join(lines)
    if sum(len(v) for v in data.values()) <= len(names):
        lines.append("  ANOVA not run: at least one group needs n >= 2.")
        return "n/a", "\n".join(lines)

    F, p = stats.f_oneway(*[data[n] for n in names])
    if np.isnan(p):
        lines.append("  ANOVA could not be calculated (all values identical).")
        return "n/a", "\n".join(lines)
    lines.append(f"  One-way ANOVA: F = {F:.3f}, p = {p:.4f}")
    if p >= 0.05:
        lines.append("  -> No significant difference between the groups (p >= 0.05).")
        return "N", "\n".join(lines)

    lines.append("  -> Significant difference (p < 0.05). Pairwise Welch t-tests, Benjamini-Hochberg adjusted p:")
    pairs, raw_p = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_, b_ = names[i], names[j]
            if len(data[a_]) < 2 or len(data[b_]) < 2:
                lines.append(f"    {a_} vs. {b_}: not tested (each group needs n >= 2)")
                continue
            _, tp = stats.ttest_ind(data[a_], data[b_], equal_var=False)
            if np.isnan(tp):
                lines.append(f"    {a_} vs. {b_}: not tested (identical values in both groups)")
                continue
            pairs.append((a_, b_)); raw_p.append(tp)
    if raw_p:
        reject, adj, _, _ = multipletests(raw_p, alpha=0.05, method='fdr_bh')
        for (a_, b_), rp, ap, sig in zip(pairs, raw_p, adj, reject):
            lines.append(f"    {a_} vs. {b_}: p = {rp:.4f}, adjusted p = {ap:.4f} ({'significant' if sig else 'not significant'})")
    return "Y", "\n".join(lines)


# =================================================================
# BUILD ALL FIGURES FROM THE CURRENT SAMPLE CONFIGURATION
# =================================================================
def build_figures(info, configs):
    """configs: list of {'name','color','idx'} already in the desired order."""
    columns = info["columns"]
    active = [t for t in TESTS if columns[t]]
    if not active:
        raise ValueError("No usable data found in these files.")

    grouped = {}
    for cfg in configs:
        grouped.setdefault(cfg['name'], []).append(cfg)
    labels = list(grouped.keys())
    n = len(labels)

    store = {"processed": {t: {} for t in TESTS},
             "rep": {"modulus": {}, "tan": {}, "yield": {}, "recovery": {}},
             "mod_bars": [], "yield_bars": [], "rec_bars": [], "notes": set()}

    # ---------------- Figure 1 ----------------
    fig1 = Figure(figsize=(6 * len(active), 6 * n))
    axes = fig1.subplots(n, len(active), squeeze=False)

    for c_idx, test in enumerate(active):
        for r_idx, label in enumerate(labels):
            cfgs = grouped[label]
            ax = axes[r_idx, c_idx]
            color = cfgs[0]['color']
            files = [info["used_files"][c['idx']] for c in cfgs if info["used_files"][c['idx']] in columns[test]]
            if not files:
                ax.axis('off')
                continue
            try:
                PLOTTERS[test](ax, info, files, label, color, store)
                ax.set_title(f"{test.upper()}\n{label}", fontweight='bold', pad=15)
                ax.grid(True, which="both", ls="-", alpha=0.15)
            except Exception:
                ax.text(0.5, 0.5, "Error Processing", ha='center', transform=ax.transAxes)
                print(f"[{test} / {label}]"); traceback.print_exc()
    fig1.tight_layout(rect=[0, 0, 1, 0.96])

    # ---------------- Figure 2 ----------------
    proc = store["processed"]
    fig2 = Figure(figsize=(22, 12))
    axs = fig2.subplots(2, 3)
    ax_t, ax_temp, ax_step = axs[0]
    ax_gp, ax_gpp, ax_amp = axs[1]

    for label, d in proc["Temperature"].items():
        for k, r in enumerate(d["reps"]):
            ax_temp.plot(r["x"], r["g1"], marker='o', ms=3, ls='-', lw=0.8, color=d["color"], alpha=0.8,
                         label=f"{label} G'" if k == 0 else None)
            ax_temp.plot(r["x"], r["g2"], marker='s', ms=3, ls='--', lw=0.8, color=d["color"], alpha=0.8,
                         label=f"{label} G''" if k == 0 else None)
    for label, d in proc["Step Shear"].items():
        ax_step.plot(d["x"], d["y"], marker='o', ls='', ms=3, alpha=0.7, color=d["color"], label=label)
        if np.any(d["s"] > 0):
            ax_step.fill_between(d["x"], d["y"] - d["s"], d["y"] + d["s"], color=d["color"], alpha=0.2, lw=0)
    for label, d in proc["Frequency"].items():
        ax_t.errorbar(d["x"], d["tan"], yerr=d["tans"], fmt='^', ls='', ms=4, label=label, color=d["color"], capsize=2)
        ax_gp.errorbar(d["x"], d["g1"], yerr=d["g1s"], fmt='o-', ms=4, label=label, color=d["color"], capsize=2)
        ax_gpp.errorbar(d["x"], d["g2"], yerr=d["g2s"], fmt='o-', ms=4, label=label, color=d["color"], capsize=2)
    for label, d in proc["Amplitude"].items():
        for k, r in enumerate(d["reps"]):
            ax_amp.plot(r["x"], r["g1"], marker='o', ms=3, ls='', color=d["color"], alpha=0.8,
                        label=f"{label} G'" if k == 0 else None)
            ax_amp.plot(r["x"], r["g2"], marker='o', ms=3, ls='', color=d["color"], mfc='none', alpha=0.8,
                        label=f"{label} G''" if k == 0 else None)

    specs = [
        (ax_t, "COMBINED TAN(δ)", 'Angular frequency (rad/s)', 'tan(δ)', True, False),
        (ax_temp, "COMBINED TEMPERATURE RAMP", "Temperature (°C)", "Pascal", False, True),
        (ax_step, "COMBINED STEP SHEAR", "Total time (s)", "Viscosity (Pa·s)", False, False),
        (ax_gp, "COMBINED G' (STORAGE)", 'Angular frequency (rad/s)', "G' (Pa)", True, True),
        (ax_gpp, "COMBINED G'' (LOSS)", 'Angular frequency (rad/s)', "G'' (Pa)", True, True),
        (ax_amp, "COMBINED AMPLITUDE SWEEP", "Strain (%)", "Pascal", True, True),
    ]
    for ax, title, xl, yl, lx, ly in specs:
        if not ax.has_data():
            ax.axis('off')
            continue
        if lx: ax.set_xscale('log')
        if ly: ax.set_yscale('log')
        ax.set_title(title, fontweight='bold'); ax.set_xlabel(xl); ax.set_ylabel(yl)
        ax.grid(True, which="both", ls="-", alpha=0.15); ax.legend()
        if ax is ax_t:
            ax.set_ylim(0, 1.5)
            ax.axhline(1.0, color=C_DEEP, ls='--', lw=ANNOTATION_WIDTH)
            ax.axhspan(0, 1.0, facecolor=C_FOREST, alpha=0.08)
        if ax is ax_step:
            ax.set_ylim(bottom=0)
    fig2.tight_layout()

    # ---------------- Statistics ----------------
    rep = store["rep"]
    runs = [("STORAGE MODULUS G' (point closest to 10 rad/s, Pa)", rep["modulus"]),
            ("TAN(DELTA) (point closest to 10 rad/s)", rep["tan"]),
            (f"YIELD STRAIN (%, G'/G'' crossover above {MIN_YIELD_STRAIN:g}%)", rep["yield"]),
            ("RECOVERY AFTER 1ST HIGH-SHEAR STEP (% of first rest step)", rep["recovery"])]
    flags, texts = [], []
    for head, data in runs:
        flag, text = run_stats(head, data)
        flags.append((head.split(" (")[0], flag)); texts.append(text)
    stats_text = "".join(f"NOTE: {m}\n" for m in sorted(store["notes"])) + ("\n" if store["notes"] else "")
    stats_text += ("Each value is one replicate (one file). Groups = rows with the exact same name.\n"
                   "Significant difference between groups? (Y = yes, N = no, n/a = not enough data)\n")
    for head, flag in flags:
        stats_text += f"  • {head}: {flag}\n"
    stats_text += "=" * 60 + "\n" + ("\n" + "-" * 60 + "\n").join(texts) + "\n"

    # ---------------- Figure 3 ----------------
    panels = [k for k, v in (("yield", store["yield_bars"]), ("mod", store["mod_bars"]), ("rec", store["rec_bars"])) if v]
    fig3 = None
    if panels:
        fig3 = Figure(figsize=(7 * len(panels), 7))
        ax3 = dict(zip(panels, fig3.subplots(1, len(panels), squeeze=False)[0]))

        if "yield" in ax3:
            ax = ax3["yield"]; yb = store["yield_bars"]
            idx = np.arange(len(yb))
            ax.bar(idx, [b["val"] for b in yb], yerr=[b["err"] for b in yb], capsize=6,
                   color=[b["color"] for b in yb], edgecolor=C_DEEP)
            ax.set_xlim(-0.5, len(yb) - 0.5)
            for i, b in enumerate(yb):
                if np.isnan(b["val"]):
                    ax.text(i, 0.5, b["note"], color='red', fontweight='bold', fontsize=10,
                            ha='center', va='center', transform=ax.get_xaxis_transform(),
                            bbox=dict(facecolor='white', alpha=0.9, edgecolor='#ffcccc', lw=1, boxstyle='round,pad=0.4'))
            ax.set_ylabel("Strain [%]"); ax.set_title("YIELD STRAIN (SEM)", fontweight='bold')
            ax.set_xticks(idx); ax.set_xticklabels([b["label"] for b in yb], rotation=45, ha='center')

        if "mod" in ax3:
            ax = ax3["mod"]; mb = store["mod_bars"]
            x = np.arange(len(mb)); width, gap = 0.32, 0.05
            cols = [b["color"] for b in mb]
            ax.bar(x - (width / 2 + gap / 2), [b["g1"] for b in mb], width, yerr=[b["g1e"] for b in mb],
                   capsize=4, label="G'", color=cols, edgecolor=C_DEEP)
            ax.bar(x + (width / 2 + gap / 2), [b["g2"] for b in mb], width, yerr=[b["g2e"] for b in mb],
                   capsize=4, label="G''", color='white', edgecolor=cols, lw=2)
            ax.set_ylabel("Pascal"); ax.set_title("MODULUS COMPARISON AT 10 RAD/S (SEM)", fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([b["label"] for b in mb], rotation=45, ha='center')
            ax.legend()

        if "rec" in ax3:
            ax = ax3["rec"]; rb = store["rec_bars"]
            n_cyc = max(len(b["mean"]) for b in rb)
            x = np.arange(len(rb)); width = 0.8 / n_cyc
            for k in range(n_cyc):
                vals = [b["mean"][k] if len(b["mean"]) > k else np.nan for b in rb]
                errs = [b["sem"][k] if len(b["sem"]) > k else 0 for b in rb]
                ax.bar(x - 0.4 + width * (k + 0.5), vals, width, yerr=errs, capsize=4,
                       color=[b["color"] for b in rb], alpha=max(0.3, 1 - 0.25 * k), edgecolor=C_DEEP)
            ax.axhline(100, color=C_DEEP, ls='--', lw=ANNOTATION_WIDTH)
            ax.set_ylabel("Recovery (% of initial viscosity)")
            ax.set_title("STRUCTURE RECOVERY (SEM)", fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels([b["label"] for b in rb], rotation=45, ha='center')
            ax.legend(handles=[Patch(facecolor='grey', edgecolor=C_DEEP, alpha=max(0.3, 1 - 0.25 * k),
                                     label=f"Cycle {k + 1}") for k in range(n_cyc)])
        fig3.tight_layout()

    return [fig1, fig2, fig3], stats_text


# =================================================================
# STREAMLIT UI
# =================================================================
def parse_color(s):
    """Accepts 'Green 3 (#4CAF50)', '#4CAF50' or a named color. Returns a valid color string or None."""
    s = str(s).strip() if s is not None else ""
    if '(' in s:
        s = s.split('(')[-1].strip(')').strip()
    return s if s and is_color_like(s) else None


def fig_bytes(fig, fmt, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


st.set_page_config(page_title="Rheology Matrix Master", page_icon="🧪", layout="wide")
st.title("🧪 Rheology Matrix Master")
st.caption("Upload dine Excel-filer (.xls / .xlsx), navngiv prøverne, og tryk på **Opdater plots**. "
           "Rækker med PRÆCIS samme navn er replikater: Frequency / Step Shear viser gennemsnit ± SEM, "
           "Amplitude / Temperature viser alle replikaters punkter.")

uploaded = st.file_uploader("Vælg Excel-filer", type=["xls", "xlsx"], accept_multiple_files=True)

if not uploaded:
    st.info("Upload en eller flere filer for at komme i gang.")
    st.stop()

file_tuple = tuple(sorted((f.name, f.getvalue()) for f in uploaded))
info = scan_uploaded(file_tuple)

if not info["used_files"]:
    st.error("Ingen Frequency-, Amplitude-, Temperature- eller Step Shear-data blev genkendt i filerne.")
    st.stop()

found = ", ".join(f"{t}: {len(v)}" for t, v in info["columns"].items() if v)
st.success(f"Fundet → {found}")

# ---------- sample table (reset when the set of files changes) ----------
signature = tuple((n, len(b)) for n, b in file_tuple)
if st.session_state.get("signature") != signature:
    st.session_state["signature"] = signature
    st.session_state["table"] = pd.DataFrame({
        "Vis": [True] * len(info["used_files"]),
        "Rækkefølge": list(range(1, len(info["used_files"]) + 1)),
        "Navn": [f"Sample {r + 1}" for r in range(len(info["used_files"]))],
        "Farve": [dropdown_labels[(r * 2) % 18] for r in range(len(info["used_files"]))],
        "Egen farve (hex)": [""] * len(info["used_files"]),
        "Fil": [f"{f}  [{', '.join(t for t in TESTS if f in info['columns'][t])}]" for f in info["used_files"]],
    })
    st.session_state.pop("results", None)

with st.form("config_form"):
    edited = st.data_editor(
        st.session_state["table"],
        hide_index=True,
        use_container_width=True,
        disabled=["Fil"],
        column_config={
            "Vis": st.column_config.CheckboxColumn("Vis", width="small"),
            "Rækkefølge": st.column_config.NumberColumn("Rækkefølge", min_value=1, step=1, width="small"),
            "Navn": st.column_config.TextColumn("Navn (samme navn = replikater)"),
            "Farve": st.column_config.SelectboxColumn("Farve", options=dropdown_labels),
            "Egen farve (hex)": st.column_config.TextColumn("Egen farve (hex)", help="Fx #FF8800 – overstyrer dropdown"),
        },
        key="editor",
    )
    submitted = st.form_submit_button("Opdater plots", type="primary")

if submitted:
    st.session_state["table"] = edited
    configs, bad = [], []
    for idx, row in edited.sort_values("Rækkefølge").iterrows():
        if not row["Vis"]:
            continue
        color = parse_color(row["Egen farve (hex)"]) or parse_color(row["Farve"])
        if color is None:
            bad.append(f"Række {idx + 1}: ugyldig farve")
            continue
        name = str(row["Navn"]).strip() or f"Sample {idx + 1}"
        configs.append({"name": name, "color": color, "idx": int(idx)})
    if bad:
        st.error("\n".join(bad))
    elif not configs:
        st.warning("Sæt kryds ved mindst én række.")
    else:
        with st.spinner("Beregner statistik og laver plots ..."):
            try:
                figs, stats_text = build_figures(info, configs)
                st.session_state["results"] = {
                    "stats": stats_text,
                    "figs": [(f, fig_bytes(f, "png"), fig_bytes(f, "pdf")) if f is not None else None for f in figs],
                }
            except Exception as e:
                traceback.print_exc()
                st.error(f"Fejl: {e}")

res = st.session_state.get("results")
if res:
    titles = ["Figur 1 – Hver prøve, hver test", "Figur 2 – Kombinerede overlays", "Figur 3 – Søjlediagrammer"]
    for i, (title, item) in enumerate(zip(titles, res["figs"])):
        if item is None:
            continue
        fig, png, pdf = item
        st.subheader(title)
        st.image(png)
        c1, c2, _ = st.columns([1, 1, 6])
        c1.download_button("Hent PNG", png, file_name=f"figur_{i + 1}.png", mime="image/png", key=f"png{i}")
        c2.download_button("Hent PDF", pdf, file_name=f"figur_{i + 1}.pdf", mime="application/pdf", key=f"pdf{i}")
    st.subheader("Statistik")
    st.code(res["stats"], language=None)
    st.download_button("Hent statistik (.txt)", res["stats"], file_name="statistik.txt", key="stats_dl")
