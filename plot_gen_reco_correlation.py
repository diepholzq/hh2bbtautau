"""
Vergleichs- und Korrelationsplots für zwei Datensätze mit awkward-Arrays.

Für jede der angegebenen Variablen (die in beiden Datensätzen unter dem
gleichen Namen vorliegen) werden erzeugt:
  1) Ein 1D-Vergleichsplot (überlagerte, normierte Histogramme)
  2) Ein 2D-Korrelationsplot (Streudiagramm / 2D-Histogramm der Werte
     aus Datensatz 1 gegen Datensatz 2, sinnvoll wenn beide Datensätze
     Event-für-Event zueinander passen, z.B. gleiche Events aus
     unterschiedlichen Rekonstruktionen)

"""

import os
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
from glob import glob


# --------------------------------------------------------------------------
# Konfiguration
# --------------------------------------------------------------------------
def load_data(parquet_file_path: str, column_name: str, is_gen: bool, is_top: bool = False):
    # check if combined file exists
    if is_top:
        try:
            events = ak.from_parquet(f"{parquet_file_path}columns_all.parquet")[column_name]
        except:
            print(f"No united  file found at {parquet_file_path}, creating one...")
            file_list = glob(f"{parquet_file_path}*.parquet")
            result = ak.concatenate([ak.from_parquet(file_list[0]), ak.from_parquet(file_list[1])], axis=0)
            column_dict = {}
            for column in result.fields:
                # ch_id_mask = result[column][result[column].fields[0]] != EMPTY_FLOAT
                ch_id_mask = result[column]["channel_id"] == 3
                column_dict[column] = result[column][ch_id_mask]
                if len(file_list) > 1:
                    for idx in range(2, len(file_list)):
                        imported = ak.from_parquet(file_list[idx])[column]
                        # ch_id_mask = imported[imported.fields[0]] != EMPTY_FLOAT
                        ch_id_mask = imported["channel_id"] == 3
                        column_dict[column] = ak.to_packed(
                            ak.concatenate([column_dict[column], imported[ch_id_mask]], axis=0)
                        )
            result = ak.zip(
                {
                    column_name: column_dict[column_name],
                }
            )
            ak.to_parquet(result, f"{parquet_file_path}columns_all.parquet")
            events = ak.from_parquet(f"{parquet_file_path}columns_all.parquet")[column_name]
        return events
    else:
        events = ak.from_parquet(f"{parquet_file_path}columns_0.parquet")[column_name]
        ch_id_mask = events["channel_id"] == 3
        data2 = {}
        for field in events.fields:
            if field != "channel_id":
                data2[field] = events[field][ch_id_mask]
        data2 = ak.zip(data2)
        return data2


# Liste der Variablennamen, die in beiden Datensätzen vorkommen
VARIABLES = [
    # tt system
    "tt_vis_system_mass",
    "tt_vis_system_pt",
    "tt_vis_system_pz",
    "tt_vis_system_phi",
    #
    "t_vis_y_diff",
    "t1_vis_phi",
    #
    "tau1_phi",
    "tau1_cos_theta_star_cms_t1_vis",
    #
    "tau2_phi",
    "tau2_cos_theta_star_cms_t2_vis",
    #
    "tau1_cos_theta_star_cms_wplus",
    "tau2_cos_theta_star_cms_wminus",
]
TICK_NAMES = [
    # tt system
    r"$m_{t_{vis}\bar{t}_{vis}}$",
    r"$p_{T, t_{vis}\bar{t}_{vis}}$",
    r"$p_{z, t_{vis} \bar{t}_{vis}}$",
    r"$\phi_{t_{vis}\bar{t}_{vis}}$",
    #
    r"$\Delta \ y(t_{vis}\bar{t}_{vis})$",
    r"$\phi_{\bar{t}_{vis}}$",
    #
    r"$\phi_{\tau^{+}}$",
    r"$cos(\theta^{*}_{\tau^{+}})^{t_{vis}}$",
    #
    r"$\phi_{\tau^{-}}$",
    r"$cos(\theta^{*}_{\tau^{-}})^{\bar{t}_{vis}}$",
    #
    r"$cos(\theta^{*}_{\tau^{+}})^{W^{+}}$",
    r"$cos(\theta^{*}_{\tau^{-}})^{W^{-}}$",
]

# Labels für die beiden Datensätze (für Legenden/Titel)
LABEL_1 = "gen"
LABEL_2 = "reco"

# Ausgabeverzeichnis für die Plots
OUTPUT_DIR = "/afs/desy.de/user/d/diepholq/Documents/Plots/event_observables/"

# Anzahl der Bins für 1D- und 2D-Histogramme
N_BINS_1D = 50
N_BINS_2D = 60


# --------------------------------------------------------------------------
# Hilfsfunktionen
# --------------------------------------------------------------------------


def to_flat_numpy(array):
    """
    Wandelt ein (ggf. jagged / nested) awkward-Array in ein flaches
    1D-numpy-Array um. Bereits flache Arrays werden direkt konvertiert.
    """
    arr = ak.Array(array)

    return ak.to_numpy(arr).astype(float)


def load_variable(dataset, name):
    """
    Holt eine Variable aus einem Datensatz (dict-artig oder awkward-Record-Array)
    und gibt sie als flaches numpy-Array zurück.
    """
    values = dataset[name]
    return to_flat_numpy(values)


def common_bin_edges(values_1, values_2, n_bins):
    """Bestimmt gemeinsame Bin-Kanten für zwei Verteilungen."""
    combined = np.concatenate([values_1, values_2])
    combined = combined[np.isfinite(combined)]
    lo, hi = np.percentile(combined, [0.05, 99.95])
    if lo == hi:
        lo, hi = combined.min(), combined.max()
    return np.linspace(lo, hi, n_bins + 1)


# --------------------------------------------------------------------------
# Plot-Funktionen
# --------------------------------------------------------------------------


def plot_1d_comparison(name, tick_name, values_1, values_2, output_dir):
    """Erstellt einen überlagerten, normierten 1D-Histogramm-Vergleichsplot."""
    bins = common_bin_edges(values_1, values_2, N_BINS_1D)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(
        values_1,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=f"{LABEL_1} (n={values_1.size})",
        color="tab:blue",
    )
    ax.hist(
        values_2,
        bins=bins,
        density=True,
        histtype="step",
        linewidth=2,
        label=f"{LABEL_2} (n={values_2.size})",
        color="tab:orange",
    )

    ax.set_xlabel(tick_name)
    ax.set_ylabel("fraction of events")
    ax.set_title(f"1D comparison: {tick_name}")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{name}_1d_comparison.pdf")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_2d_correlation(name, tick_name, values_1, values_2, output_dir):
    """
    Erstellt einen 2D-Korrelationsplot (2D-Histogramm) von Datensatz 1
    gegen Datensatz 2 für die gleiche Variable. Setzt voraus, dass beide
    Arrays die gleiche Länge haben (Event-für-Event-Zuordnung).
    """
    if values_1.size != values_2.size:
        n = min(values_1.size, values_2.size)
        print(f"[Warnung] '{name}': unterschiedliche Länge " f"({values_1.size} vs {values_2.size}). Kürze auf {n}.")
        values_1 = values_1[:n]
        values_2 = values_2[:n]

    combined = np.concatenate([values_1, values_2])
    combined = combined[np.isfinite(combined)]
    lo, hi = np.percentile(combined, [0.5, 99.5])
    if lo == hi:
        lo, hi = combined.min(), combined.max()

    fig, ax = plt.subplots(figsize=(6, 6))

    h = ax.hist2d(
        values_1,
        values_2,
        bins=N_BINS_2D,
        range=[[lo, hi], [lo, hi]],
        cmap="viridis",
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="number of entries")

    # Diagonale y = x als Referenzlinie
    ax.plot([lo, hi], [lo, hi], color="red", linestyle="--", linewidth=1, label="y = x")

    # Korrelationskoeffizient berechnen (auf gültigen, endlichen Werten)
    mask = np.isfinite(values_1) & np.isfinite(values_2)
    if mask.sum() > 1:
        corr = np.corrcoef(values_1[mask], values_2[mask])[0, 1]
    else:
        corr = float("nan")

    ax.set_xlabel(f"{tick_name} ({LABEL_1})")
    ax.set_ylabel(f"{tick_name} ({LABEL_2})")
    ax.set_title(f"2D correlation: {tick_name}  (r = {corr:.3f})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out_path = os.path.join(output_dir, f"{name}_2d_correlation.pdf")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# Hauptfunktion
# --------------------------------------------------------------------------


def create_comparison_plots(dataset_1, dataset_2, variables=VARIABLES, output_dir=OUTPUT_DIR):
    """
    Erstellt für jede Variable in `variables` einen 1D-Vergleichsplot und
    einen 2D-Korrelationsplot zwischen dataset_1 und dataset_2.

    Parameter
    ---------
    dataset_1, dataset_2 : dict oder awkward.Array (Record-Array)
        Müssen per Variablennamen indizierbar sein, z.B. dataset_1["tau1_phi"].
    variables : list[str]
        Liste der zu vergleichenden Variablennamen.
    output_dir : str
        Zielverzeichnis für die erzeugten PDF-Dateien.
    """
    os.makedirs(output_dir, exist_ok=True)

    for idx, name in enumerate(variables):
        print(f"Verarbeite Variable: {name}")
        try:
            values_1 = load_variable(dataset_1, name)
            values_2 = load_variable(dataset_2, name)
        except Exception as exc:
            print(f"  [Fehler] Konnte '{name}' nicht laden: {exc}")
            continue

        if values_1.size == 0 or values_2.size == 0:
            print(f"  [Warnung] '{name}' ist in einem Datensatz leer, übersprungen.")
            continue

        path_1d = plot_1d_comparison(name, TICK_NAMES[idx], values_1, values_2, output_dir)
        print(f"  -> 1D-Plot gespeichert: {path_1d}")

        path_2d = plot_2d_correlation(name, TICK_NAMES[idx], values_1, values_2, output_dir)
        print(f"  -> 2D-Plot gespeichert: {path_2d}")

    print(f"\nFertig. Alle Plots liegen in: {os.path.abspath(output_dir)}")


# --------------------------------------------------------------------------
# Beispielhafter Aufruf
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # HIER die eigenen Datensätze laden, z.B. mit uproot/awkward aus ROOT-
    # oder Parquet-Dateien:
    #
    #   import uproot
    #   dataset_1 = uproot.open("file1.root")["Events"].arrays(VARIABLES)
    #   dataset_2 = uproot.open("file2.root")["Events"].arrays(VARIABLES)
    #
    # oder aus Parquet-Dateien:
    #
    #   dataset_1 = ak.from_parquet("dataset1.parquet")
    #   dataset_2 = ak.from_parquet("dataset2.parquet")
    #
    # Anschließend:
    #
    #   create_comparison_plots(dataset_1, dataset_2)
    # dataset 1 : gen
    # dataset 2: reco
    parquet_file_path_top_reco = "/data/dust/user/diepholq/hh2bbtautau/correlation_data/ttbar/reco/"
    parquet_file_path_top_gen = "/data/dust/user/diepholq/hh2bbtautau/correlation_data/ttbar/gen/"
    dataset_1 = load_data(parquet_file_path_top_gen, "pdf_input_vars_gen_top", True, is_top=True)
    dataset_2 = load_data(parquet_file_path_top_reco, "pdf_input_vars_reco_top", False, is_top=True)
    create_comparison_plots(dataset_1, dataset_2)
