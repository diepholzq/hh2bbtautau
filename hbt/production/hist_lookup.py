from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
import hbt.production.likelihood_helper_functions as lhf
from hbt.production.pdf_input_vars_reco import pdf_inputs
from columnflow.columnar_util import EMPTY_FLOAT

ak = maybe_import("awkward")
np = maybe_import("numpy")


@producer(
    # uses={"pdf_input_vars_reco_higgs.*", "pdf_input_vars_reco_top.*"},
    uses={pdf_inputs},
    produces={"likelihood_ratio"},
)
def calculate_likelihood_ratio(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """Given the signal and background likelihood inputs, evaluates events as both likelihood, then calculates the
    ratio. Also takes into account the jacobian determinant and mass constraint terms
    """
    events = self[pdf_inputs](events, **kwargs)
    n_bins_1d = 750
    bins_per_dim_2d = np.array([27, 27])
    bins_per_dim_3d = np.array([20, 7, 7])

    # Evaluate provided events as both likelihoods
    eval_data_signal = events.pdf_input_vars_reco_higgs
    p_is_higgs_log, err_is_higgs, err_is_higgs_log, ev_idx_higgs = lhf.calculate_reco_score_higgs(
        eval_data_signal,
        n_bins_1d,
        bins_per_dim=bins_per_dim_2d,
        allow_hist_creation=False,
        statistical_binning=True,
        do_constr=True,
        bin_filling=False,
        do_jacobian=True,
    )
    eval_data_bg = events.pdf_input_vars_reco_top
    p_is_top_log, err_is_top, err_is_top_log, ev_idx_top = lhf.calculate_reco_score_top(
        eval_data_bg,
        n_bins_1d,
        bins_per_dim=bins_per_dim_3d,
        allow_hist_creation=False,
        statistical_binning=True,
        do_constr=True,
        bin_filling=False,
        do_jacobian=True,
    )
    ev_idx_mask = ev_idx_higgs == ev_idx_top
    p_is_higgs, err_is_higgs = p_is_higgs_log[ev_idx_mask], err_is_higgs[ev_idx_mask]
    p_is_top, err_is_top = p_is_top_log[ev_idx_mask], err_is_top[ev_idx_mask]

    ratio = p_is_higgs - p_is_top
    # errors_ratio = np.sqrt((1 / np.exp(p_is_higgs) * err_is_higgs)**2 + (-1 / np.exp(p_is_top) * err_is_top)**2)

    # Make column full length again
    ev_idx = ev_idx_higgs[ev_idx_mask]
    all_idx = np.arange(0, len(events), 1)
    event_mask = np.isin(all_idx, ev_idx)
    ratio_column = np.ones(len(events)) * EMPTY_FLOAT
    ratio_column[event_mask] = ratio
    events = set_ak_column(events, "likelihood_ratio", ratio_column)
    return events
