from hbt.production.histogram_helper_functions import get_data
import numpy as np

import matplotlib.pyplot as plt
from sympy import Matrix, pprint


def log_and_remove_nans(data: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.log(data), nan=-99999)[np.nan_to_num(np.log(data), nan=-99999) != -99999]


def plot_jac_det(
    higgs_path: str,
    top_path: str,
    save_fig: bool = False,
    plot_path: str = None,
    plot_name: str = None,
) -> None:
    # get jacobian terms
    # from IPython import embed
    #
    # embed(header="plot jac det ")
    higgs_is_higgs_data = get_data(higgs_path, "pdf_input_vars_reco_higgs", drop_nones=True)
    higgs_is_top_data = get_data(higgs_path, "pdf_input_vars_reco_top", drop_nones=True)
    top_is_top_data = get_data(top_path, "pdf_input_vars_reco_top", drop_nones=True)
    top_is_higgs_data = get_data(top_path, "pdf_input_vars_reco_higgs", drop_nones=True)

    # calc. log
    higgs_is_higgs_det, top_is_top_det = log_and_remove_nans(abs(higgs_is_higgs_data.jac_det)), log_and_remove_nans(
        abs(top_is_top_data.jac_det)
    )
    higgs_is_top_det, top_is_higgs_det = log_and_remove_nans(abs(higgs_is_top_data.jac_det)), log_and_remove_nans(
        abs(top_is_higgs_data.jac_det)
    )

    # set edges for plotting
    min_edge_higgs = min(np.percentile(higgs_is_higgs_det, 0.01), np.percentile(top_is_higgs_det, 0.01))
    max_edge_higgs = max(np.percentile(higgs_is_higgs_det, 99.99), np.percentile(higgs_is_top_det, 99.99))
    max_edge_top = max(np.percentile(top_is_higgs_det, 99.99), np.percentile(top_is_top_det, 99.99))
    min_edge_top = min(np.percentile(top_is_higgs_det, 0.01), np.percentile(top_is_top_det, 0.01))

    # Plotting of jacobian terms
    fig, ax = plt.subplots(2, 1, figsize=(9, 9))
    ax[0].hist(
        (higgs_is_higgs_det, higgs_is_top_det),
        bins=np.linspace(min_edge_higgs, max_edge_higgs, 100),
        density=True,
        label=(r"$log$($J(L^{HH})$)", r"$log$($J(L^{t\bar{t}})$)"),
        histtype="step",
        color=("b", "r"),
        linewidth=2,
    )
    ax[1].hist(
        (top_is_higgs_det, top_is_top_det),
        bins=np.linspace(min_edge_top, max_edge_top, 100),
        density=True,
        label=("$log$($J(L^{HH})$)", r"$log$($J(L^{t\bar{t}})$)"),
        histtype="step",
        color=("b", "r"),
        linewidth=2,
    )
    ax[0].legend(), ax[1].legend()
    ax[0].set_title(r"Signal Dataset")
    ax[1].set_title(r"Background Dataset")
    if save_fig:
        if not plot_name:
            plot_name = "jac_det_comparison"
        plt.savefig(f"{plot_path}{plot_name}.pdf")
    else:
        plt.show()
    plt.close()

    # Plotting of per-event differences of jacobian terms
    signal_data_det_diff = higgs_is_higgs_det - higgs_is_top_det
    bg_data_det_diff = top_is_higgs_det - top_is_top_det
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))
    ax[0].hist(
        signal_data_det_diff,
        bins=100,
        label=r"$log$($J(L^{HH})$) - $log$($J(L^{t\bar{t}})$)",
        histtype="step",
        color="b",
        linewidth=2,
    )
    ax[1].hist(
        bg_data_det_diff,
        bins=100,
        label=r"$log$($J(L^{HH})$) - $log$($J(L^{t\bar{t}})$)",
        histtype="step",
        color="r",
        linewidth=2,
    )
    ax[2].hist(
        (signal_data_det_diff, bg_data_det_diff),
        bins=100,
        density=True,
        label=(r"$\Delta J$ Signal dataset", r"$\Delta J$ BG Dataset"),
        histtype="step",
        color=("b", "r"),
        linewidth=2,
    )
    ax[0].legend(), ax[1].legend(), ax[2].legend()
    ax[0].set_title(r"Signal Dataset")
    ax[1].set_title(r"Background Dataset")
    plot_name_diff = "jac_det_differences"
    if save_fig:
        plt.savefig(f"{plot_path}{plot_name_diff}.pdf")
    else:
        plt.show()
    plt.close()


def print_jac_matrix_examples(matrix: np.ndarray) -> None:
    matrix
    matrix_str = Matrix(matrix)
    matrix_str = Matrix([[f"{elem:.2f}" for elem in row] for row in matrix_str.tolist()])
    pprint(matrix_str)


if __name__ == "__main__":
    # PATH_HIGGS = (
    #     "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
    #     "hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/prod24/columns_0.parquet"
    # )
    # PATH_TOP = (
    #     "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
    #     "tt_dl_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/prod24/columns_all.parquet"
    # )
    PATH_HIGGS_1BOOST = "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22post_v14/hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/pdf_inputs_1boosts/"
    PATH_HIGGS_2BOOST = "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22post_v14/hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/pdf_inputs_2boosts/"
    PATH_TOP_1BOOST = "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22post_v14/tt_dl_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/pdf_inputs_1boosts/"
    PATH_TOP_2BOOST = "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22post_v14/tt_dl_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/pdf_inputs_2boosts/"
    PLOT_PATH = "/afs/desy.de/user/d/diepholq/Documents/Plots/jacobians/"
    plot_jac_det(PATH_HIGGS_2BOOST, PATH_TOP_2BOOST, save_fig=True, plot_path=PLOT_PATH)
