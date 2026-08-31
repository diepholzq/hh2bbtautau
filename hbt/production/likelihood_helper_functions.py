"""Calculates per-event likelihood scores for different likelihood creation methods. Returns them together
with statistical uncertainties.
"""

import hbt.production.histogram_helper_functions as hhf
from columnflow.util import maybe_import
from columnflow.columnar_util import EMPTY_FLOAT

ak = maybe_import("awkward")
np = maybe_import("numpy")

LINES_STRING = (
    "------------------------------------------------------------------------------------------------------\n"
)


def log_err_prop(res_dict: dict, p_string, err_string):
    """Gaussian error propagation for log likelihood"""
    res = np.sqrt(((1 / res_dict[p_string]) * res_dict[err_string]) ** 2)
    return res


def calculate_reco_score_higgs(
    eval_data: ak.Array,
    n_bins_1d: int,
    bins_per_dim: np.ndarray = np.array([27, 27]),
    allow_hist_creation: bool = False,
    statistical_binning: bool = True,
    do_constr: bool = True,
    return_log: bool = True,
    bin_filling: bool = False,
    do_jacobian: bool = False,
    check_files_higgs: bool = True,
    check_files_top: bool = True,
):
    """Calculates likelihood scores for events being signal-like.
    Args:
        eval_data_path (str): Path to the parquet file containing the pdf_input_vars
        n_bins_1d (int): Number of bins for 1d factorization steps
        bins_per_dim (np.ndarray): Array with the bins for the 2d factorization step
        allow_hist_creations (bool): Behavior if histogram for chosen options does not exist
        statistical_binning (bool): If the statistical_binning tweak should be used
        do_constr (bool): If constraint likelihood terms should be considered
        return_log (bool): If the probabilities should be returned as ln(probabilities)
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        do_jacobian (bool): If jacobian determinant of transformation from detector to pdf input space should be taken
        check_files_higgs (bool): If existence of combined parquet files should be checked
        check_files_top (bool): If existence of combined parquet files should be checked
                            into account
    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray): Array with the event probabilities, Array with the statistical
        uncertainties per event, event indices that survived none dropping
    """
    constr_b: ak.Array = eval_data.constr_term_b
    constr_tau: ak.Array = eval_data.constr_term_tau

    ev_idx: np.ndarray = np.arange(0, len(eval_data), 1)
    # from IPython import embed
    # embed(header="calc reco score higgs")
    # non_msk: ak.Array = ak.is_none(eval_data) == bool(0)
    non_msk: ak.Array = eval_data[eval_data.fields[0]] != EMPTY_FLOAT

    # ignore jacobian terms if not desired
    if do_jacobian:
        jac_det: ak.Array = np.abs(eval_data.jac_det[non_msk])
    else:
        jac_det = np.ones_like(ev_idx)[non_msk]

    eval_data, constr_b, constr_tau, ev_idx = (
        eval_data[non_msk],
        constr_b[non_msk],
        constr_tau[non_msk],
        ev_idx[non_msk],
    )

    higgs_results_dict = hhf.eval_higgs_likelihood_hists(
        eval_data,
        n_bins_1d,
        bins_per_dim=bins_per_dim,
        allow_hist_creation=allow_hist_creation,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        bin_filling=bin_filling,
        check_files_higgs=check_files_higgs,
    )
    probs = higgs_results_dict["probs_higgs"]
    errors = higgs_results_dict["errors_higgs"]
    if do_constr:
        if not return_log:
            probs = probs * np.exp(-(constr_b + constr_tau)) * jac_det
            return probs, errors, ev_idx
        else:
            errors_log = np.sqrt(((1 / probs) * errors) ** 2)
            probs = np.log(probs) - constr_b - constr_tau + np.log(jac_det)
            return probs, errors, errors_log, ev_idx
    else:
        if return_log:
            errors_log = np.sqrt(((1 / probs) * errors) ** 2)
            probs = np.log(probs)
            return probs, errors, errors_log, ev_idx
        else:
            return probs, errors, ev_idx


def calculate_reco_score_top(
    eval_data: ak.Array,
    n_bins_1d: int,
    bins_per_dim: np.ndarray = np.array([20, 7, 7]),
    allow_hist_creation: bool = False,
    statistical_binning: bool = True,
    do_constr: bool = True,
    return_log: bool = True,
    bin_filling: bool = False,
    do_jacobian: bool = False,
    check_files_higgs: bool = True,
    check_files_top: bool = True,
):
    """Calculates likelihood scores for events being background-like.
    Args:
        eval_data_path (str): Path to the parquet file containing the pdf_input_vars
        n_bins_1d (int): Number of bins for 1d factorization steps
        bins_per_dim (np.ndarray): Array with the bins for the 3d factorization step
        allow_hist_creations (bool): Behavior if histogram for chosen options does not exist
        statistical_binning (bool): If the statistical_binning tweak should be used
        do_constr (bool): If constraint likelihood terms should be considered
        return_log (bool): If likelihoods should be transformed to log scale
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        do_jacobian (bool): If jacobian determinant of transformation from detector to pdf input space should be taken
                            into account
        check_files_higgs (bool): If existence of combined parquet files should be checked
        check_files_top (bool): If existence of combined parquet files should be checked
    Returns:
        Tuple(np.ndarray, np.ndarray): Array with the event probabilities, Array with the statistical uncertainties per
        event, event indices that survived none dropping. If return_log, linear as well as log errors are returned
    """
    ev_idx: np.ndarray = np.arange(0, len(eval_data), 1)

    non_msk: ak.Array = eval_data[eval_data.fields[0]] != EMPTY_FLOAT
    # non_msk: ak.Array = ak.is_none(eval_data) == bool(0)
    print(f"do jacobian: {do_jacobian}")
    # ignore jacobian terms if not desired
    if do_jacobian:
        jac_det: ak.Array = np.abs(eval_data.jac_det[non_msk])
    else:
        jac_det = np.ones_like(ev_idx)[non_msk]

    eval_data, ev_idx = eval_data[non_msk], ev_idx[non_msk]

    top_results_dict = hhf.eval_top_likelihood_hists(
        eval_data,
        n_bins_1d,
        bins_per_dim=bins_per_dim,
        allow_hist_creation=allow_hist_creation,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        bin_filling=bin_filling,
        check_files_top=check_files_top,
    )
    probs, errors = top_results_dict["p_top"], top_results_dict["errors_p_top"]
    if return_log:
        errors_log = np.sqrt(((1 / probs) * errors) ** 2)
        probs = np.log(probs) + np.log(jac_det)
        return probs, errors, errors_log, ev_idx
    else:
        return probs * jac_det, errors, ev_idx


def calculate_likelihood_ratio_for_plotting(
    eval_data_path: str,
    n_bins_1d: int = 750,
    bins_per_dim_2d: np.ndarray = np.array([27, 27]),
    bins_per_dim_3d: np.ndarray = np.array([20, 7, 7]),
    statistical_binning: bool = True,
    allow_hist_creation: bool = False,
    do_constr: bool = True,
    do_jacobian: bool = False,
    bin_filling: bool = False,
    check_files_higgs: bool = True,
    check_files_top: bool = True,
):
    """Calculates the log likelihood ratio for a set of events, the events need to have the columns
    pdf_input_vars_reco_higgs and pdf_input_vars_reco_top.
    Args:
        eval_data_path (str): Path to the file containing the events
        n_bins_1d (int): Number of bins for 1d factorization steps
        bins_per_dim_2d (np.ndarray): Array with the bins for the 2d factorization step
        bins_per_dim_3d (np.ndarray): Array with the bins for the 3d factorization step
        statistical_binning (bool): If the statistical_binning tweak should be used
        do_constr (bool): If the gen constraint term should be considered
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        do_jacobian (bool): If jacobian determinant of transformation from detector to pdf input space should be taken
                            into account
        check_files_higgs (bool): If existence of combined parquet files should be checked
        check_files_top (bool): If existence of combined parquet files should be checked
    Returns:
        Tuple(np.ndarray, np.ndarray): Array with the event likelihood ratios, Array with the statistical uncertainties
        per event
    """
    # TODO: add all options or remove
    print(
        f"\n\n{LINES_STRING}"
        f"Calculating likelihood ratio, with the following options:\neval_data_path: {eval_data_path}\n"
        f"n_bins_1d: {n_bins_1d}\nstatistical_binning: {statistical_binning}\ndo_constr: {do_constr}"
        f"\ndo_jacobian: {do_jacobian}:"
        f"\n{LINES_STRING}\n",
    )
    eval_data_signal = hhf.get_data(eval_data_path, "pdf_input_vars_reco_higgs", drop_nones=True)
    p_is_higgs_log, err_is_higgs, err_is_higgs_log, ev_idx_higgs = calculate_reco_score_higgs(
        eval_data_signal,
        n_bins_1d,
        bins_per_dim=bins_per_dim_2d,
        allow_hist_creation=allow_hist_creation,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        bin_filling=bin_filling,
        do_jacobian=do_jacobian,
        check_files_higgs=check_files_higgs,
        check_files_top=check_files_top,
    )

    eval_data_bg = hhf.get_data(eval_data_path, "pdf_input_vars_reco_top", drop_nones=True)
    p_is_top_log, err_is_top, err_is_top_log, ev_idx_top = calculate_reco_score_top(
        eval_data_bg,
        n_bins_1d,
        bins_per_dim=bins_per_dim_3d,
        allow_hist_creation=allow_hist_creation,
        statistical_binning=statistical_binning,
        do_constr=do_constr,
        bin_filling=bin_filling,
        do_jacobian=do_jacobian,
        check_files_higgs=check_files_higgs,
        check_files_top=check_files_top,
    )

    ev_idx_mask = ev_idx_higgs == ev_idx_top
    p_is_higgs, err_is_higgs = p_is_higgs_log[ev_idx_mask], err_is_higgs[ev_idx_mask]
    p_is_top, err_is_top = p_is_top_log[ev_idx_mask], err_is_top[ev_idx_mask]

    ratio = p_is_higgs - p_is_top
    errors_ratio = np.sqrt((1 / np.exp(p_is_higgs) * err_is_higgs) ** 2 + (-1 / np.exp(p_is_top) * err_is_top) ** 2)

    return ratio, errors_ratio
