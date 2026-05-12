import hbt.production.histogram_helper_functions as hhf
import hbt.production.likelihood_helper_functions as lhf
from columnflow.columnar_util import EMPTY_FLOAT, maybe_import, EMPTY_INT

import matplotlib.pyplot as plt
from sympy import Matrix, pprint
from sklearn.metrics import roc_curve, roc_auc_score    # , RocCurveDisplay
from operator import itemgetter
from glob import glob

ak = maybe_import("awkward")
np = maybe_import("numpy")


def match_events(ev_idx_a: np.array, ev_idx_b: np.array) -> list:
    """Given the event indices of two likelihood distributions (may have different lenghts, this function returns
    an array with indices that, applied to distribution b, return all the events that correspond to the events in
    distribution a, so that the two have the same length and correspond element for element.
    Args:
        ev_idx_a (np.array): Array containing event indices for distribution a
        ev_idx_b (np.array): Array containing event indices for distribution b
    Returns:
        sorted_indices (np.array): Array with indices as described above
    """
    sorted_indices_a: np.array = np.arange(0, len(ev_idx_a), 1)
    sorted_indices_b: np.array = np.zeros_like(ev_idx_a)
    # Match indices from a to b
    for idx, ev_idx in enumerate(ev_idx_a):
        if len(np.ravel(np.nonzero(ev_idx_b == ev_idx))) > 0:
            sorted_indices_b[idx] = np.ravel(np.nonzero(ev_idx_b == ev_idx))[0]
        else:
            sorted_indices_b[idx] = EMPTY_INT
    # Delete non-matched events
    empty_mask = sorted_indices_b != EMPTY_INT
    sorted_indices_a = sorted_indices_a[empty_mask]
    sorted_indices_b = sorted_indices_b[empty_mask]
    return [sorted_indices_a, sorted_indices_b]


def remove_nones(likelihood_distribution: np.array) -> np.array:
    """Removes None, posinf, neginf from likelihood_distribution array
    """
    likelihood_distribution = np.nan_to_num(
        likelihood_distribution, nan=EMPTY_FLOAT, posinf=EMPTY_FLOAT, neginf=EMPTY_FLOAT
    )
    likelihood_distribution = ak.to_packed(likelihood_distribution[likelihood_distribution != EMPTY_FLOAT])
    return likelihood_distribution


def calculate_roc_values(ratio_higgs_cleaned: np.ndarray, ratio_top_cleaned: np.ndarray) -> dict:
    """Calculates roc curve and auc of likelihood ratio classifier + uncertainty on the roc auc
    Args:
        ratio_higgs_cleaned (np.ndarray): L-ratio for signal class
        ratio_top_cleaned (np.ndarray): L-ratio for background class
    Returns:
        roc_values (dict): Dict with signal sens, bg rej., auc score, uncert. on auc
    """
    roc_auc_std = hhf.roc_auc_std

    max_score = np.max([np.max(ratio_higgs_cleaned), np.max(ratio_top_cleaned)])
    ratio_higgs_cleaned = ratio_higgs_cleaned / max_score
    ratio_top_cleaned = ratio_top_cleaned / max_score
    n_p = len(ratio_higgs_cleaned)
    n_n = len(ratio_top_cleaned)
    y_score = np.concatenate([ratio_higgs_cleaned, ratio_top_cleaned], axis=0)
    y_score = ak.to_packed(y_score)
    y_true = np.concatenate([np.ones_like(ratio_higgs_cleaned), 0 * np.ones_like(ratio_top_cleaned)], axis=0)
    y_true = ak.to_packed(y_true)
    roc_auc = roc_auc_score(y_true, y_score)
    roc_auc_std = roc_auc_std(roc_auc, n_p, n_n)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    signal_sens = tpr
    bg_rej = 1 - fpr

    roc_values = {
        "signal_sens": signal_sens,
        "bg_rej": bg_rej,
        "roc_auc": roc_auc,
        "roc_auc_std": roc_auc_std,
    }
    return roc_values


def print_correlation_matrix(eval_data_path: str, column_name: str, do_constr: bool = True) -> None:
    """Calculates Pearson correlation coefficient between all inputs and prints corresponding matrix
    Args:
        eval_data_path (str): Path to the parquet file containing the pdf_input_vars
        column_name (str): Name of the pdf inputs column, used to distinguish between higgs and top
        do_constr (bool): If the gen constraint term should be considered
    """
    plot_dir = "/afs/desy.de/user/d/diepholq/Documents/Plots/correlation_plots/"
    pdf_inputs = hhf.get_data(parquet_file_path=eval_data_path, column_name=column_name, drop_nones=True)
    higgs_inputs = [
        "dihiggs_mass",
        "dihiggs_system_pt",
        "dihiggs_system_pz",
        "dihiggs_system_phi",
        "cos_theta_h1",
        "phi_h1",
        "cos_theta_cms_h2_tau_vis1",
        "phi_cms_h2_tau_vis1",
        "cos_theta_cms_h1_b1",
        "phi_cms_h1_b1",
    ]
    higgs_ticks = [
        # r"$m_{inv}(H_{bb}, H_{\tau\tau})$",
        r"$m_{HH}$",
        # r"$p_{T}(H_{bb}, H_{\tau\tau})$",
        r"$p_{T, HH}$",
        # r"$p_{z}(H_{bb}, H_{\tau\tau})$",
        r"$p_{z, HH}$",
        r"$\phi_{HH}$",
        # r"$cos(\theta(H_1^{HH}))$",
        r"$cos(\theta^{*}_{H_{bb}})$",
        # r"$\phi(H_1^{HH})$",
        r"$\phi_{H_{bb}}$",
        # r"$cos(\theta(\tau_{vis,1}^{H_2})$",
        r"$cos(\theta^{*}_{H_{\tau\tau}})$",
        # r"$\phi(\tau_{vis,1}^{H_2})$",
        r"$\phi_{H_{\tau\tau}}$",
        # r"$cos(\theta(b_1^{H_1}))$",
        r"$cos(\theta^{*}_{b})$",
        # r"$\phi(b_1^{H_1})$",
        r"$\phi_{b}$",
    ]
    top_inputs = [
        "tt_vis_system_mass",
        "tt_vis_system_pt",
        "tt_vis_system_pz",
        "tt_vis_system_phi",
        "t_vis_y_diff",
        "t1_vis_phi",
        "tau1_cos_theta_star_cms_t1_vis",
        "tau1_phi",
        "tau2_cos_theta_star_cms_t2_vis",
        "tau2_phi",
        "tau1_cos_theta_star_cms_wplus",
        "tau2_cos_theta_star_cms_wminus",
    ]
    top_ticks = [
        r"$m_{t_{vis}\bar{t}_{vis}}$",
        r"$p_{T, t_{vis}\bar{t}_{vis}}$",
        r"$p_{z, t_{vis} \bar{t}_{vis}}$",
        r"$\phi_{t_{vis}\bar{t}_{vis}}$",
        r"$\Delta \ y(t_{vis}\bar{t}_{vis})$",
        r"$\phi_{\bar{t}_{vis}}$",
        r"$cos(\theta^{*}_{\tau^{+}})^{\bar{t}_{vis}}$",
        r"$\phi_{\tau^{+}}$",
        r"$cos(\theta^{*}_{\tau^{-}})^{t_{vis}}$",
        r"$\phi_{\tau^{-}}$",
        r"$cos(\theta^{*}_{\tau^{+}})^{W^{+}}$",
        r"$cos(\theta^{*}_{\tau^{.}})^{W^{+}}$",
    ]

    if np.any(column_name.split("_") == np.full_like(column_name.split("_"), "higgs", dtype=f"<U{len('higgs')}")):
        inputs = higgs_inputs
        ticks = higgs_ticks
    elif np.any(column_name.split("_") == np.full_like(column_name.split("_"), "top", dtype=f"<U{len('top')}")):
        inputs = top_inputs
        ticks = top_ticks
    else:
        raise Exception("Not detected if signal or background column was provided")
    coef_arr = np.empty((len(inputs), len(pdf_inputs)))
    for idx, input in enumerate(inputs):
        coef_arr[idx] = eval(f"pdf_inputs.{input}")

    # print matrix
    corrcoef_matrix = np.corrcoef(coef_arr)
    # for better printing:
    corrcoef_matrix_str = Matrix(corrcoef_matrix)
    corrcoef_matrix_str = Matrix([[f"{elem:.2f}" for elem in row] for row in corrcoef_matrix_str.tolist()])
    pprint(corrcoef_matrix_str)

    # Plot Matrix
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corrcoef_matrix)
    fig.colorbar(mappable=im)
    ax.set_yticks(np.arange(0, len(inputs), 1), labels=ticks)
    ax.set_xticks(np.arange(0, len(inputs), 1), labels=ticks, rotation="vertical")
    for x_idx, row in enumerate(corrcoef_matrix_str.tolist()):
        for y_idx, elem in enumerate(row):
            ax.text(x_idx, y_idx, f"{elem:.2f}", ha="center", va="center", color="w")
    plt.savefig(
        f"{plot_dir}{eval_data_path.split('/')[-1].split('.')[0]}__{column_name}__do_constr-{do_constr}.pdf", dpi=300
    )
    plt.close()


def get_likelihoods_and_ratios(
    eval_data_path_higgs: str,
    eval_data_path_top: str,
    n_bins_1d: int,
    bins_per_dim_2d: np.ndarray,
    bins_per_dim_3d: np.ndarray,
    allow_hist_creations: bool = False,
    statistical_binning: bool = True,
    do_constr: bool = True,
    bin_filling: bool = False,
    calculate_uncertainties: bool = True,
    do_jacobian: bool = False,
) -> dict:
    """Given the higgs and ttbar dataset paths, evaluates events as signal and background likelihoods,
    calculates ratios and returns individual likelihoods as well as ratios.
    Args:
        eval_data_path_higgs (str): Path to the parquet file containing the pdf_input_vars of the signal dataset
        eval_data_path_top (str): Path to the parquet file containing the pdf_input_vars of the bg dataset
        n_bins_1d (int): Number of bins for 1d factorization steps
        bins_per_dim_2d (np.ndarray): Array with the bins for the 2d factorization step
        bins_per_dim_3d (np.ndarray): Array with the bins for the 3d factorization step
        allow_hist_creations (bool): Behavior if histogram for chosen options does not exist
        statistical_binning (bool): If the statistical_binning tweak should be used
        do_constr (bool): If the gen constraint term should be considered
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        calculate_uncertainties (bool): If statistical uncertainties should be calculated
        do_jacobian (bool): If jacobian determinant of transformation from detector to pdf input space should be taken
                            into account
    Returns:
        likelihood_dict (dict): Dictionary containing the likelihoods and ratios
    """
    calculate_likelihood_ratio = lhf.calculate_likelihood_ratio_for_plotting
    calculate_reco_score_top = lhf.calculate_reco_score_top
    calculate_reco_score_higgs = lhf.calculate_reco_score_higgs

    # Calculate ratios
    ratio_higgs_log, errors_ratio_higgs_log = calculate_likelihood_ratio(
        eval_data_path_higgs,
        n_bins_1d=n_bins_1d,
        bins_per_dim_2d=bins_per_dim_2d,
        bins_per_dim_3d=bins_per_dim_3d,
        statistical_binning=statistical_binning,
        allow_hist_creation=allow_hist_creations,
        do_constr=do_constr,
        bin_filling=bin_filling,
        do_jacobian=do_jacobian,
    )
    ratio_top_log, errors_ratio_top_log = calculate_likelihood_ratio(
        eval_data_path_top,
        n_bins_1d=n_bins_1d,
        bins_per_dim_2d=bins_per_dim_2d,
        bins_per_dim_3d=bins_per_dim_3d,
        statistical_binning=statistical_binning,
        allow_hist_creation=allow_hist_creations,
        do_constr=do_constr,
        bin_filling=bin_filling,
        do_jacobian=do_jacobian,
    )

    # Evaluate individual likelihoods for both datasets:
    # Evaluate higgs events on higgs likelihood
    data_higgs_is_higgs = hhf.get_data(eval_data_path_higgs, "pdf_input_vars_reco_higgs", drop_nones=False)
    p_higgs_is_higgs_log, err_higgs_is_higgs, err_higgs_is_higgs_log, ev_idx_higgs = calculate_reco_score_higgs(
        data_higgs_is_higgs,
        n_bins_1d,
        bins_per_dim=bins_per_dim_2d,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        return_log=True,
        do_jacobian=do_jacobian,
    )
    # Evaluate higgs events on top likelihood
    data_higgs_is_top = hhf.get_data(eval_data_path_higgs, "pdf_input_vars_reco_top", drop_nones=False)
    p_higgs_is_top_log, err_higgs_is_top, err_higgs_is_top_log, ev_idx_top = calculate_reco_score_top(
        data_higgs_is_top,
        n_bins_1d,
        bins_per_dim=bins_per_dim_3d,
        statistical_binning=statistical_binning,
        return_log=True,
        do_jacobian=do_jacobian,
    )
    # ...
    data_top_is_higgs = hhf.get_data(eval_data_path_top, "pdf_input_vars_reco_higgs", drop_nones=False)
    p_top_is_higgs_log, err_top_is_higgs, err_top_is_higgs_log, ev_idx_higgs = calculate_reco_score_higgs(
        data_top_is_higgs,
        n_bins_1d,
        bins_per_dim=bins_per_dim_2d,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        return_log=True,
        do_jacobian=do_jacobian,
    )
    data_top_is_top = hhf.get_data(eval_data_path_top, "pdf_input_vars_reco_top", drop_nones=False)
    p_top_is_top_log, err_top_is_top, err_top_is_top_log, ev_idx_top = calculate_reco_score_top(
        data_top_is_top,
        n_bins_1d,
        bins_per_dim=bins_per_dim_3d,
        statistical_binning=statistical_binning,
        return_log=True,
        do_jacobian=do_jacobian,
    )

    likelihood_dict = {
        "ratio_higgs_log": ratio_higgs_log,
        "errors_ratio_higgs_log": errors_ratio_higgs_log,
        "ratio_top_log": ratio_top_log,
        "errors_ratio_top_log": errors_ratio_top_log,
        "p_higgs_is_higgs_log": p_higgs_is_higgs_log,
        "err_higgs_is_higgs_log": err_higgs_is_higgs_log,
        "p_higgs_is_top_log": p_higgs_is_top_log,
        "err_higgs_is_top_log": err_higgs_is_top_log,
        "p_top_is_higgs_log": p_top_is_higgs_log,
        "err_top_is_higgs_log": err_top_is_higgs_log,
        "p_top_is_top_log": p_top_is_top_log,
        "err_top_is_top_log": err_top_is_top_log,
    }

    return likelihood_dict


def plot_performance_metrics(
    eval_data_path_higgs: str,
    eval_data_path_top: str,
    n_bins_1d: int,
    bins_per_dim2d: np.ndarray = np.array([27, 27]),
    bins_per_dim3d: np.ndarray = np.array([20, 7, 7]),
    n_mc: int = 300,
    allow_hist_creations: bool = False,
    statistical_binning: bool = True,
    do_constr: bool = True,
    bin_filling: bool = False,
    calculate_uncertainties: bool = True,
    do_jacobian: bool = True,
    plot_identifier: str = "",
    plot_dir: str = "",
) -> None:
    """Plots a set of performance metrics: The roc curve, roc auc value, the log likelihood ratios, the
    ROC auc + uncertainties as function of n_bins.
    Tweaks such as statistical aware binning can be turned on/off.
    Args:
        eval_data_path_higgs (str): Path to the parquet file containing the pdf_input_vars of the signal dataset
        eval_data_path_top (str): Path to the parquet file containing the pdf_input_vars of the bg dataset
        n_bins_1d (int): Number of bins for 1d factorization steps
        bins_per_dim_2d (np.ndarray): Array with the bins for the 2d factorization step
        bins_per_dim_3d (np.ndarray): Array with the bins for the 3d factorization step
        n_mc (int): Number of Monte Carlo throws for the stat. uncert. calc.
        statistical_binning (bool): If the statistical_binning tweak should be used
        do_constr (bool): If the gen constraint term should be considered
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        calculate_uncertainties (bool): If statistical uncertainties should be calculated
        do_jacobian (bool): If jacobian determinant of transformation from detector to pdf input space should be taken
                            into account
        plot_identifier (str): String to add to plot name to identify it later
        plot_dir (str): String with directory where plots should be stored
    """
    plot_dir = plot_dir
    likelihood_plot_name = (
        f"likelihood_plots_{plot_identifier}_{n_bins_1d}_stat-bin-{statistical_binning}"
        f"_do-constr-{do_constr}"
        f"_bin-filling-{bin_filling}"
        f"_uncertainties-{calculate_uncertainties}"
        f"_jacobian-{do_jacobian}.pdf"
    )

    likelihood_dict = get_likelihoods_and_ratios(
        eval_data_path_higgs,
        eval_data_path_top,
        n_bins_1d,
        bins_per_dim2d,
        bins_per_dim3d,
        allow_hist_creations=allow_hist_creations,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        bin_filling=bin_filling,
        calculate_uncertainties=calculate_uncertainties,
        do_jacobian=do_jacobian,
    )

    (ratio_higgs_log,
    errors_ratio_higgs_log,
    ratio_top_log,
    errors_ratio_top_log,
    p_higgs_is_higgs_log,
    err_higgs_is_higgs_log,
    p_higgs_is_top_log,
    err_higgs_is_top_log,
    p_top_is_higgs_log,
    err_top_is_higgs_log,
    p_top_is_top_log,
    err_top_is_top_log) = itemgetter(
        "ratio_higgs_log",
        "errors_ratio_higgs_log",
        "ratio_top_log",
        "errors_ratio_top_log",
        "p_higgs_is_higgs_log",
        "err_higgs_is_higgs_log",
        "p_higgs_is_top_log",
        "err_higgs_is_top_log",
        "p_top_is_higgs_log",
        "err_top_is_higgs_log",
        "p_top_is_top_log",
        "err_top_is_top_log")(likelihood_dict)

    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    roc_plot_name = (
        f"roc_plot_{plot_identifier}_{n_bins_1d}_stat-bin-{statistical_binning}"
        f"_do-constr-{do_constr}"
        f"_bin-filling-{bin_filling}"
        f"_uncertainties-{calculate_uncertainties}"
        f"_jacobian-{do_jacobian}.png"
    )
    if calculate_uncertainties:
        smeared_ratio_higgs = np.random.normal(loc=ratio_higgs_log[:, None],
                                            scale=errors_ratio_higgs_log[:, None],
                                            size=(len(ratio_higgs_log), n_mc)).T
        smeared_ratio_top = np.random.normal(loc=ratio_top_log[:, None],
                                            scale=errors_ratio_top_log[:, None],
                                            size=(len(ratio_top_log), n_mc)).T

        roc_auc_i = np.zeros(n_mc, dtype=np.float64)
        roc_std_i = np.zeros(n_mc, dtype=np.float64)
        for idx in range(n_mc):
            progress_percentage = int((idx / n_mc) * 100)          # progress-o-meter
            num_hashes = int(progress_percentage)
            print(f"mc uncertainty estimation in progress - roc: [{'#' * num_hashes}{' ' * (100 - num_hashes)}] \
                {progress_percentage}% completed", end="\r")
            ratio_higgs_cleaned = np.nan_to_num(
                smeared_ratio_higgs[idx], nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
            )
            # print(f"\nsampledratio higgs cleaned nans: {len(ratio_higgs_cleaned[ratio_higgs_cleaned == EMPTY_INT])}\n")
            ratio_higgs_cleaned = ak.to_packed(ratio_higgs_cleaned[ratio_higgs_cleaned != EMPTY_INT])
            ratio_top_cleaned = np.nan_to_num(
                smeared_ratio_top[idx], nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
            )
            # print(f"\nsampledratio top cleaned nans: {len(ratio_top_cleaned[ratio_top_cleaned == EMPTY_INT])}\n")
            ratio_top_cleaned = ak.to_packed(ratio_top_cleaned[ratio_top_cleaned != EMPTY_INT])

            roc_values_i = calculate_roc_values(ratio_higgs_cleaned, ratio_top_cleaned)

            signal_sens, bg_rej = roc_values_i["signal_sens"], roc_values_i["bg_rej"]
            roc_auc_i[idx]: np.float64 = roc_values_i["roc_auc"]
            roc_std_i[idx]: np.float64 = roc_values_i["roc_auc_std"]
            ax_roc.plot(signal_sens, bg_rej, linestyle="None", marker=".")

        mean_auc = np.mean(roc_auc_i)
        std_auc = np.std(roc_auc_i)

        print(f"\n\n\nauc score: {mean_auc:.4f}, with std dev: {std_auc:.4f}.\n\
        Standard error according to formula: {np.mean(roc_std_i)}")

        ratio_higgs_cleaned_nom = np.nan_to_num(
            ratio_higgs_log, nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
        )
        n_sig_events = len(ratio_higgs_cleaned_nom) - len(ratio_higgs_cleaned_nom[ratio_higgs_cleaned_nom == EMPTY_INT])
        print(f"\nnom ratio higgs cleaned nans: {len(ratio_higgs_cleaned_nom[ratio_higgs_cleaned_nom == EMPTY_INT])}\n")
        ratio_higgs_cleaned_nom = ak.to_packed(ratio_higgs_cleaned_nom[ratio_higgs_cleaned_nom != EMPTY_INT])
        ratio_top_cleaned_nom = np.nan_to_num(
            ratio_top_log, nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
        )
        print(f"\nnom ratio top cleaned nans: {len(ratio_top_cleaned_nom[ratio_top_cleaned_nom == EMPTY_INT])}\n")
        n_bg_events = len(ratio_top_cleaned_nom) - len(ratio_top_cleaned_nom[ratio_top_cleaned_nom == EMPTY_INT])
        ratio_top_cleaned_nom = ak.to_packed(ratio_top_cleaned_nom[ratio_top_cleaned_nom != EMPTY_INT])

        roc_values_nom = calculate_roc_values(ratio_higgs_cleaned_nom, ratio_top_cleaned_nom)
        signal_sens_nom, bg_rej_nom = roc_values_nom["signal_sens"], roc_values_nom["bg_rej"]
        roc_auc_nom = roc_values_nom["roc_auc"]
        # roc_std_nom = roc_values_nom["roc_auc_std"]

        ax_roc.plot(
            signal_sens_nom,
            bg_rej_nom,
            label=f"ROC auc: {roc_auc_nom:.4f}",
            linestyle="--", color="black", linewidth="1.5",
        )
        ax_roc.legend()
        ax_roc.set_ylabel("Signal sensitivity")
        ax_roc.set_xlabel("Background rejection")
        plusminus = r"$\pm$"
        ax_roc.set_title(f"Mean auc: {mean_auc:.4f}{plusminus}{std_auc:.4f}")
        roc_plot_text = (
            f"stat-bin: {statistical_binning}\n"
            f"do-constr: {do_constr}\n"
            f"bin_filling: {bin_filling}\n"
            f"Number of signal events: {n_sig_events}\n"
            f"Number of background events: {n_bg_events}"
        )
        ax_roc.text(0.07, 0.07, roc_plot_text)
    else:
        ratio_higgs_cleaned = np.nan_to_num(
            ratio_higgs_log, nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
        )
        ratio_higgs_cleaned = ak.to_packed(ratio_higgs_cleaned[ratio_higgs_cleaned != EMPTY_INT])
        ratio_top_cleaned = np.nan_to_num(
            ratio_top_log, nan=EMPTY_INT, posinf=EMPTY_INT, neginf=EMPTY_INT,
        )
        ratio_top_cleaned = ak.to_packed(ratio_top_cleaned[ratio_top_cleaned != EMPTY_INT])
        roc_values_nom = calculate_roc_values(ratio_higgs_cleaned, ratio_top_cleaned)
        signal_sens_nom, bg_rej_nom = roc_values_nom["signal_sens"], roc_values_nom["bg_rej"]
        roc_auc_nom = roc_values_nom["roc_auc"]
        # roc_std_nom = roc_values_nom["roc_auc_std"]

        ax_roc.plot(
            signal_sens_nom,
            bg_rej_nom,
            label=f"ROC auc: {roc_auc_nom:.4f}",
            linestyle="--", color="black", linewidth="1.5",
        )
        ax_roc.set_ylabel("Signal sensitivity")
        ax_roc.set_xlabel("background rejection")
        ax_roc.legend()
    print(f"saving {plot_dir}{roc_plot_name}...")
    plt.savefig(f"{plot_dir}{roc_plot_name}", dpi=300)
    print("done :)")
    plt.close()

    # Likelihood and ratio comparison plots
    fig_l, ax_l = plt.subplot_mosaic(
        [["sig", "bg"],
        ["ratios", "ratios"]],
        gridspec_kw={"wspace": 0.3},
        figsize=(9, 9),
    )
    if calculate_uncertainties:
        create_smeared_hists = hhf.create_smeared_hists
        # Likelihood plots - mc smearing
        (mean_higgs_is_higgs,
        std_higgs_is_higgs,
        bin_centers_higgs_is_higgs,
        bin_width_higgs_is_higgs) = create_smeared_hists(
            p_higgs_is_higgs_log,
            err_higgs_is_higgs_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [0.5])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )
        (mean_higgs_is_top,
        std_higgs_is_top,
        bin_centers_higgs_is_top,
        bin_width_higgs_is_top) = create_smeared_hists(
            p_higgs_is_top_log,
            err_higgs_is_top_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [0.5])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )
        (mean_top_is_higgs,
        std_top_is_higgs,
        bin_centers_top_is_higgs,
        bin_width_top_is_higgs) = create_smeared_hists(
            p_top_is_higgs_log,
            err_top_is_higgs_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [0.5])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )
        (mean_top_is_top,
        std_top_is_top,
        bin_centers_top_is_top,
        bin_width_top_is_top) = create_smeared_hists(
            p_top_is_top_log,
            err_top_is_top_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [0.5])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )

        # Prepare data for plt.stairs
        edges_higgs_is_higgs = np.concatenate([bin_centers_higgs_is_higgs - bin_width_higgs_is_higgs /
        2, [bin_centers_higgs_is_higgs[-1] + bin_width_higgs_is_higgs / 2]])
        edges_higgs_is_top = np.concatenate([bin_centers_higgs_is_top - bin_width_higgs_is_top /
            2, [bin_centers_higgs_is_top[-1] + bin_width_higgs_is_top / 2]])
        edges_top_is_higgs = np.concatenate([bin_centers_top_is_higgs - bin_width_top_is_higgs /
            2, [bin_centers_top_is_higgs[-1] + bin_width_top_is_higgs / 2]])
        edges_top_is_top = np.concatenate([bin_centers_top_is_top - bin_width_top_is_top /
            2, [bin_centers_top_is_top[-1] + bin_width_top_is_top / 2]])

        # Likelihood plots:
        ax_l["sig"].bar(bin_centers_higgs_is_higgs,
                    mean_higgs_is_higgs,
                    width=bin_width_higgs_is_higgs,
                    yerr=std_higgs_is_higgs,
                    color="none",
                    edgecolor="none",
                    alpha=0.6,
                    ecolor="b")
        ax_l["sig"].stairs(mean_higgs_is_higgs, edges_higgs_is_higgs, color="b", label=r"$H$ dataset")
        ax_l["sig"].bar(bin_centers_top_is_higgs,
                    mean_top_is_higgs,
                    width=bin_width_top_is_higgs,
                    yerr=std_top_is_higgs,
                    color="none",
                    edgecolor="none",
                    alpha=0.6,
                    ecolor="r")
        ax_l["sig"].stairs(mean_top_is_higgs, edges_top_is_higgs, color="r", label=r"$t\bar{t}$ dataset")
        ax_l["sig"].set_title(r"$H$ dataset")
        ax_l["sig"].set_title(r"$\text{L}(HH\rightarrow bb\tau\tau)$")
        ax_l["sig"].legend()
        ax_l["bg"].bar(bin_centers_higgs_is_top,
                    mean_higgs_is_top,
                    width=bin_width_higgs_is_top,
                    yerr=std_higgs_is_top,
                    color="none",
                    edgecolor="none",
                    alpha=0.6,
                    ecolor="b")
        ax_l["bg"].stairs(mean_higgs_is_top, edges_higgs_is_top, color="b", label=r"$H$ dataset")
        ax_l["bg"].bar(bin_centers_top_is_top,
                    mean_top_is_top,
                    width=bin_width_top_is_top,
                    yerr=std_top_is_top,
                    color="none",
                    edgecolor="none",
                    alpha=0.6,
                    ecolor="r")
        ax_l["bg"].stairs(mean_top_is_top, edges_top_is_top, color="r", label=r"$t\bar{t}$ dataset")
        ax_l["bg"].set_title(r"$\text{L}(t\bar{t})$")
        ax_l["bg"].legend()

        # Ratio plots
        mean_ratio_higgs, std_ratio_higgs, bin_centers_ratio_higgs, bin_width_ratio_higgs = create_smeared_hists(
            ratio_higgs_log,
            errors_ratio_higgs_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [1])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )
        mean_ratio_top, std_ratio_top, bin_centers_ratio_top, bin_width_ratio_top = create_smeared_hists(
            ratio_top_log,
            errors_ratio_top_log,
            n_mc,
            bins="np.linspace(np.percentile(counts[counts < 99999], [1])[0], \
                            np.percentile(counts[counts < 99999], [99])[0], 300)",
        )

        # Preparation for plt.stairs:
        edges_ratio_higgs = np.concatenate([bin_centers_ratio_higgs - bin_width_ratio_higgs /
        2, [bin_centers_ratio_higgs[-1] + bin_width_ratio_higgs / 2]])
        edges_ratio_top = np.concatenate([bin_centers_ratio_top - bin_width_ratio_top /
        2, [bin_centers_ratio_top[-1] + bin_width_ratio_top / 2]])

        ax_l["ratios"].bar(bin_centers_ratio_higgs,
                    mean_ratio_higgs,
                    width=bin_width_ratio_higgs,
                    yerr=std_ratio_higgs,
                    color="none",
                    edgecolor="none",
                    ecolor="b",
                    alpha=0.5)
        ax_l["ratios"].stairs(mean_ratio_higgs, edges_ratio_higgs, color="b", label=r"$H$ dataset")
        ax_l["ratios"].bar(bin_centers_ratio_top,
                    mean_ratio_top,
                    width=bin_width_ratio_top,
                    yerr=std_ratio_top,
                    color="none",
                    edgecolor="none",
                    ecolor="r",
                    alpha=0.5)
        ax_l["ratios"].stairs(mean_ratio_top, edges_ratio_top, color="r", label=r"$t\bar{t}$ dataset")
    else:
        p_higgs_is_higgs_log = remove_nones(p_higgs_is_higgs_log)
        ax_l["sig"].hist(
            p_higgs_is_higgs_log,
            bins=np.linspace(np.percentile(p_higgs_is_higgs_log, 0.01), np.percentile(p_higgs_is_higgs_log, 99.99), 200),
            histtype="step", density=True,
            linewidth=2,
            alpha=0.7,
            color="b",
            label=r"$H$ dataset",
        )
        p_top_is_higgs_log = remove_nones(p_top_is_higgs_log)
        ax_l["sig"].hist(
            p_top_is_higgs_log,
            bins=np.linspace(np.percentile(p_top_is_higgs_log, 0.01), np.percentile(p_top_is_higgs_log, 99.99), 200),
            histtype="step", density=True,
            linewidth=2,
            alpha=0.7,
            color="r",
            label=r"$t\bar{t}$ dataset",
        )
        p_higgs_is_top_log = remove_nones(p_higgs_is_top_log)
        ax_l["bg"].hist(
            p_higgs_is_top_log,
            bins=np.linspace(np.percentile(p_higgs_is_top_log, 0.01), np.percentile(p_higgs_is_top_log, 99.99), 200),
            histtype="step", density=True,
            linewidth=2,
            alpha=0.7,
            color="b",
            label=r"$H$ dataset",
        )
        p_top_is_top_log = remove_nones(p_top_is_top_log)
        ax_l["bg"].hist(
            p_top_is_top_log,
            bins=np.linspace(np.percentile(p_top_is_top_log, 0.01), np.percentile(p_top_is_top_log, 99.99), 200),
            histtype="step", density=True,
            linewidth=2,
            alpha=0.7,
            color="r",
            label=r"$t\bar{t}$ dataset",
        )
        ax_l["ratios"].hist(
            ratio_higgs_cleaned,
            bins=np.linspace(np.percentile(ratio_higgs_cleaned, 0.01), np.percentile(ratio_higgs_cleaned, 99.99), 200),
            density=True,
            label=r"$H$ dataset",
            histtype="step",
            linewidth=2,
            color="b",
        )
        ax_l["ratios"].hist(
            ratio_top_cleaned,
            bins=np.linspace(np.percentile(ratio_top_cleaned, 0.01), np.percentile(ratio_top_cleaned, 99.99), 200),
            density=True,
            label=r"$t\bar{t}$ dataset",
            histtype="step",
            linewidth=2,
            color="r",
        )
    for ax_name in ["sig", "bg", "ratios"]:
        ax_l[ax_name].legend()
    ax_l["sig"].set_xlabel(r"$log(L^{HH})$")
    ax_l["sig"].set_ylabel("Fraction of events")
    ax_l["bg"].set_xlabel(r"$log(L^{t\bar{t}})$")
    ax_l["bg"].set_ylabel("Fraction of events")
    ax_l["ratios"].set_xlabel(r"$log(\frac{\text{L}^{HH}}{\text{L}^{t\bar{t}}}$)")
    ax_l["ratios"].set_ylabel("Fraction of events")
    # ax_l["ratios"].set_title(r"$\frac{\text{L}(HH\rightarrow bb\tau\tau)}{\text{L}(t\bar{t})}$")
    print(f"Saving likelihood plots to: {plot_dir}{likelihood_plot_name}...")
    plt.savefig(f"{plot_dir}{likelihood_plot_name}")
    print("Done")
    plt.close()


def plot_performance_metrics_wrapper(
    parquet_file_path_signal: str,
    parquet_file_path_bg: str,
    plot_dir: str,
) -> None:
    # General options
    n_bins_1d: int = 750
    bins_per_dim2d: np.ndarray = np.array([27, 27])
    bins_per_dim3d: np.ndarray = np.array([20, 7, 7])

    # Make sure files exist
    try:
        _ = hhf.get_data(
            f"{parquet_file_path_bg}columns_all.parquet", "pdf_input_vars_reco_top", drop_nones=True,
        )
    except:
        file_list: list = glob(f"{parquet_file_path_bg}*.parquet")
        result: ak.Array = ak.concatenate(
            [ak.from_parquet(file_list[0]),
            ak.from_parquet(file_list[1])], axis=0)
        for idx in range(2, len(file_list)):
            result = ak.concatenate([result, ak.from_parquet(file_list[idx])], axis=0)
            ak.to_parquet(result, f"{parquet_file_path_bg}columns_all.parquet")
    finally:
        path_top: str = f"{parquet_file_path_bg}columns_all.parquet"

    path_higgs: str = f"{parquet_file_path_signal}columns_0.parquet"

    # Plotting
    plot_performance_metrics(
        path_higgs,
        path_top,
        n_bins_1d,
        bins_per_dim2d=bins_per_dim2d,
        bins_per_dim3d=bins_per_dim3d,
        n_mc=300,
        allow_hist_creations=True,
        statistical_binning=True,
        do_constr=True,
        bin_filling=False,
        calculate_uncertainties=False,
        plot_dir="/afs/desy.de/user/d/diepholq/Documents/Plots/performance_metrics/",
        do_jacobian=False,
        # plot_dir="/tmp/",
    )


if __name__ == "__main__":
    PATH_HIGGS = (
        "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
        "hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/"
        "dev_likelihood_ratio/"
    )

    PATH_TOP = (
        "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/tt_dl_powheg/"
        "nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"
    )
    plot_performance_metrics_wrapper(
        PATH_HIGGS,
        PATH_TOP,
        "/afs/desy.de/user/d/diepholq/Documents/Plots/performance_metrics/",
    )
    print_correlation_matrix(PATH_HIGGS, "pdf_input_vars_reco_higgs")
    print_correlation_matrix(PATH_TOP, "pdf_input_vars_reco_top")
