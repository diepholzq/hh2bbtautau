#! /Users/quint/.venv3.11/bin/python

import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from typing import Tuple


def get_data(parquet_file_path: str) -> ak.Array:
    """Extracts the relevant fields from input data (meaning signal genlvl MC)
    Args:
        parquet_file_path (str): The path where the parquet input file is located
    Returns:
        data (ak.Array): An ak.Array containing the relevant fields
    """
    objects = ak.from_parquet(parquet_file_path)
    data = objects.pdf_input_vars
    return data


def create_any_hist(
    data: ak.Array, field_list: np.ndarray, bins: int = 10
) -> Tuple[np.ndarray, list, float]:
    """Build a multidimensional histogram from the input data and the fields provided, with a given number of bins per
    dimension
    Args:
        data (ak.Array): The data to fill the histogram
        field_list (numpy.ndarray): A list containing the field names which will make up the dimensions of the
        histogram as strings
        bins (int): The number of bins per dimension
    Returns:
        Tuple(np.ndarray, list): The nd-histogram and a list of np.ndarray containing the edges for each dimension
    """
    # Build coordinate_list as input for hist
    coordinate_list = np.empty((len(data), len(field_list)))
    idx = 0
    for field in field_list:
        coordinate_list[:, idx] = data[field]
        idx += 1

    # Create hist
    hist, edges = np.histogramdd(coordinate_list, bins=bins, density=True)
    n_dims = len(field_list)
    bin_width_dict = {}
    for dim in range(n_dims):
        bin_width_dict[f"dim_{dim}"] = edges[dim][2] - edges[dim][1]
    for dim in range(n_dims):
        if dim == 0:
            bin_volume = bin_width_dict[f"dim_{dim}"]
        else:
            bin_volume = bin_volume * bin_width_dict[f"dim_{dim}"]  # type: ignore
    print(
        f"bin_volume: {bin_volume},\n bin_volume * np.sum(hist) = {bin_volume * np.sum(hist)}"  # type: ignore
    )
    # Normalization per hist
    # smth like: for bin in hist (bin in edges (?)):
    #   bin = bin/(sum(bins) * bin_volume)
    return hist, edges, bin_volume  # type:ignore


def get_event_likelihood(
    events: ak.Array,
    hist: np.ndarray,
    edges: list,
    fields: np.ndarray,
    bin_nr: int = 10,
) -> np.ndarray:
    """Evaluates the likelihood (hist) on the provided events. Used for evaluating one factorization step
    Args:
        events (ak.Array): The data the likelihood is to be evaluated on
        hist (np.ndarray): The multidimensional histogram
        edges (list): The list of np.ndarrays containing the bin edges along each dimension
        fields (list): The fields of data
        bin_nr (int): Number of bins per dimension
    Returns:
        hist (np.ndarray): The evaluated histogram, i.e. the likelihood score for each event
    """
    # normalize hist:
    # hist = hist / np.sum(hist)
    # Normalize histogram and divide by the product of bin widths (for density)
    # bin_widths = np.array([edges[i][1] - edges[i][0] for i in range(len(edges))])
    # bin_volume = np.prod(bin_widths)
    # print(bin_volume)
    # hist = hist / np.prod(bin_widths)
    # print(np.sum(hist))

    # get bin indices for event
    num_fields = len(fields)
    if len(fields) == 0:
        raise ValueError("No fields provided")

    bin_indices = np.empty((len(events), num_fields))
    idx = 0
    for field in fields:
        bin_indices[:, idx] = np.digitize(events[field], edges[idx][1:], right=True)  # type: ignore
        idx += 1

    bin_indices[bin_indices == bin_nr] = bin_nr - 1
    bin_indices = bin_indices.astype(dtype=np.int32)
    bin_indices = bin_indices.T

    # return bin values for each event
    return hist[tuple(bin_indices)]


def top_likelihood(eval_data, bin_nr: int):
    data = get_data(
        "/Users/quint/Documents/Studium/Hiwi_Master/files/columns_0_top.parquet"
    )
    hist_s_hat, edges_s_hat, _ = create_any_hist(
        data, field_list=np.array(["s_hat_system_pt", "s_hat_system_pz", "s_hat", "s_hat_system_phi"]))

    hist_t0, edges_t0, _ = create_any_hist(
        data, field_list=np.array(["theta_t", "phi_t"]))

    hist_W_t0, edges_W_t0, _ = create_any_hist(
        data, field_list=np.array(["W_t0_theta", "W_t0_phi"]))

    hist_W_t1, edges_W_t1, _ = create_any_hist(
        data, field_list=np.array(["W_t1_theta", "W_t1_phi"]))

    hist_W_t0_child, edges_W_t0_child, _ = create_any_hist(
        data, field_list=np.array(["W_t0_child_theta", "W_t0_child_phi"]))

    hist_W_t1_child, edges_W_t1_child, _ = create_any_hist(
        data, field_list=np.array(["W_t1_child_theta", "W_t1_child_phi"]))

    p_s_hat = get_event_likelihood(
        eval_data,
        hist_s_hat,
        edges_s_hat,
        fields=np.array(["s_hat_system_pt", "s_hat_system_pz", "s_hat", "s_hat_system_phi"]),
        bin_nr=bin_nr
    )

    p_t0 = get_event_likelihood(
        eval_data,
        hist_t0,
        edges_t0,
        fields=np.array(["theta_t", "phi_t"]),
        bin_nr=bin_nr,
    )

    p_W_t0 = get_event_likelihood(
        eval_data,
        hist_W_t0,
        edges_W_t0,
        fields=np.array(["W_t0_theta", "W_t0_phi"]),
        bin_nr=bin_nr
    )

    p_W_t1 = get_event_likelihood(
        eval_data,
        hist_W_t1,
        edges_W_t1,
        fields=np.array(["W_t1_theta", "W_t1_phi"]),
        bin_nr=bin_nr
    )

    p_W_t0_child = get_event_likelihood(
        eval_data,
        hist_W_t0_child,
        edges_W_t0_child,
        fields=np.array(["W_t0_child_theta", "W_t0_child_phi"]),
        bin_nr=bin_nr
    )

    p_W_t1_child = get_event_likelihood(
        eval_data,
        hist_W_t1_child,
        edges_W_t1_child,
        fields=np.array(["W_t1_child_theta", "W_t1_child_phi"]),
        bin_nr=bin_nr
    )

    p_top = p_s_hat * p_t0 * p_W_t0 * p_W_t1 * p_W_t0_child * p_W_t1_child

    errors_s_hat = np.sqrt(p_s_hat)
    errors_t0 = np.sqrt(p_t0)
    errors_W_t0 = np.sqrt(p_W_t0)
    errors_W_t1 = np.sqrt(p_W_t1)
    errors_W_t0_child = np.sqrt(p_W_t0_child)
    errors_W_t1_child = np.sqrt(p_W_t1_child)

    errors_p_top = np.sqrt(
        (p_t0 * p_W_t0 * p_W_t1 * p_W_t0_child * p_W_t1_child * errors_s_hat) ** 2
        + (p_s_hat * p_W_t0 * p_W_t1 * p_W_t0_child * p_W_t1_child * errors_t0) ** 2
        + (p_s_hat * p_t0 * p_W_t1 * p_W_t0_child * p_W_t1_child * errors_W_t0) ** 2
        + (p_s_hat * p_t0 * p_W_t0 * p_W_t0_child * p_W_t1_child * errors_W_t1) ** 2
        + (p_s_hat * p_t0 * p_W_t0 * p_W_t1 * p_W_t1_child * errors_W_t0_child) ** 2
        + (p_s_hat * p_t0 * p_W_t0 * p_W_t1 * p_W_t0_child * errors_W_t1_child) ** 2
    )

    return p_top, errors_p_top


def higgs_likelihood(eval_data: ak.Array, bin_nr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates the "signal" likelihood and evaluates it with the given data
    Args:
        eval_data (ak.Array): Data to evaluate the likelihood on
        bin_nr (int): Number of bins for each n-d hist
    Returns:
        Tuple(np.ndarry, np.ndarray): Array with the event probabilities, Array with the statistical uncertainties per
        event
    """
    # Create likelihood:
    data = get_data(
        parquet_file_path="/Users/quint/Documents/Studium/Hiwi_Master/files/columns_0_higgs.parquet"
    )

    # These should now be normalized
    hist_higgs_system, edges_higgs_system, bin_volume_higgs_system = create_any_hist(
        data,
        field_list=np.array(
            [
                "dihiggs_mass",
                "dihiggs_system_phi",
                "dihiggs_system_pt",
                "dihiggs_system_pz",
            ]
        ),
        bins=bin_nr,
    )
    hist_higgs_h1, edges_higgs_h1, bin_volume_h1 = create_any_hist(
        data, field_list=np.array(["cos_theta_h_1", "phi_star_h_1"]), bins=bin_nr
    )
    hist_higgs_decay_h1, edges_higgs_decay_h1, bin_volume_decay_h1 = create_any_hist(
        data,
        field_list=np.array(["cos_theta_cms_h1_b1", "phi_cms_h_1_b_1"]),
        bins=bin_nr,
    )
    hist_higgs_decay_h2, edges_higgs_decay_h2, bin_volume_decay_h2 = create_any_hist(
        data,
        field_list=np.array(["cos_theta_cms_h_2_tau_1", "phi_cms_h_2_tau_1"]),
        bins=bin_nr,
    )

    # Get "probabilities"
    probs_higgs_system = get_event_likelihood(
        eval_data,
        hist_higgs_system,
        edges_higgs_system,
        fields=np.array(
            [
                "dihiggs_mass",
                "dihiggs_system_phi",
                "dihiggs_system_pt",
                "dihiggs_system_pz",
            ]
        ),
        bin_nr=bin_nr,
    )
    print()
    probs_higgs_h1 = get_event_likelihood(
        eval_data,
        hist_higgs_h1,
        edges_higgs_h1,
        fields=np.array(["cos_theta_h_1", "phi_star_h_1"]),
        bin_nr=bin_nr,
    )
    probs_higgs_decay_h1 = get_event_likelihood(
        eval_data,
        hist_higgs_decay_h1,
        edges_higgs_decay_h1,
        fields=np.array(["cos_theta_cms_h1_b1", "phi_cms_h_1_b_1"]),
        bin_nr=bin_nr,
    )
    probs_higgs_decay_h2 = get_event_likelihood(
        eval_data,
        hist_higgs_decay_h2,
        edges_higgs_decay_h2,
        fields=np.array(["cos_theta_cms_h_2_tau_1", "phi_cms_h_2_tau_1"]),
        bin_nr=bin_nr,
    )

    print(
        "np.sum(probs_higgs_system),\
        np.sum(probs_higgs_h1),\
        np.sum(probs_higgs_decay_h1),\
        np.sum(probs_higgs_decay_h2),"
    )
    print(
        np.sum(probs_higgs_system) * bin_volume_higgs_system,
        np.sum(probs_higgs_h1) * bin_volume_h1,
        np.sum(probs_higgs_decay_h1) * bin_volume_decay_h1,
        np.sum(probs_higgs_decay_h2) * bin_volume_decay_h2,
    )

    errors_higgs_system = np.sqrt(probs_higgs_system)
    errors_higgs_h1 = np.sqrt(probs_higgs_h1)
    errors_higgs_decay_h1 = np.sqrt(probs_higgs_decay_h1)
    errors_higgs_decay_h2 = np.sqrt(probs_higgs_decay_h2)

    probs_higgs = (
        probs_higgs_system
        * probs_higgs_h1
        * probs_higgs_decay_h1
        * probs_higgs_decay_h2
    )
    print(np.sum(probs_higgs))
    errors_higgs = np.sqrt(
        (
            probs_higgs_h1
            * probs_higgs_decay_h1
            * probs_higgs_decay_h2
            * errors_higgs_system
        )
        ** 2
        + (
            probs_higgs_system
            * probs_higgs_decay_h1
            * probs_higgs_decay_h2
            * errors_higgs_h1
        )
        ** 2
        + (
            probs_higgs_system
            * probs_higgs_h1
            * probs_higgs_decay_h2
            * errors_higgs_decay_h1
        )
        ** 2
        + (
            probs_higgs_system
            * probs_higgs_h1
            * probs_higgs_decay_h1
            * errors_higgs_decay_h2
        )
        ** 2
    )
    # errors_higgs = errors_higgs*probs_higgs

    return probs_higgs, errors_higgs


def create_smeared_hists(
    counts: np.ndarray,
    errors: np.ndarray,
    n_mc: int,
    bins: str = "np.linspace(0,25,26)",
    # normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Adds uncertainties to the final likelihood using MC smearing.
    Args:
        counts (np.ndarray): The final likelihood score (histogram counts) for each event
        errors (np.ndarray): The uncertainties on the counts
        n_mc (int): Number of times the MC is generated
        bins (str): An expression describing the binning, to be evaluated
    Returns:
        Tuple(np.ndarray, np.ndarray, np.ndarray, float): mean values of the smeared histograms, standard deviation on
        those, the bin centers, the bin width

    """
    # from IPython import embed
    # embed(header="create_smeared_hists")
    counts = np.nan_to_num(counts)
    errors = np.nan_to_num(errors)
    smeared = np.random.normal(
        loc=counts[:, None], scale=errors[:, None], size=(len(counts), n_mc)
    )
    smeared = smeared.T
    max_val = np.max(smeared)
    # Generate hists from samples
    all_hists: np.ndarray = np.array(
        [
            np.histogram(smeared_i[smeared_i >= 0], bins=eval(f"np.linspace(0, {max_val}, 70)"))[0]
            for smeared_i in smeared
        ]
    )

    edges: np.ndarray = eval(bins)
    hist_mean: np.ndarray = np.mean(all_hists, axis=0)
    hist_std: np.ndarray = np.std(all_hists, axis=0)  # / hist_mean
    bin_width: float = edges[1] - edges[0]
    bin_centers: np.ndarray = edges[:-1] + bin_width / 2
    # if normalize:
    #     hist_mean: np.ndarray = hist_mean / np.sum(hist_mean)
    #     hist_std: np.ndarray = hist_std * hist_mean
    return hist_mean, hist_std, bin_centers, bin_width


def plot_likelihood(probs, uncertainty, savefig=False):
    # exit(0)
    from IPython import embed

    embed(header="plot_likelihood")
    hist, std_dev, bin_centers, bin_width = create_smeared_hists(
        probs,
        uncertainty,
        n_mc=300,
        bins="np.linspace(0,np.max(smeared),70)",
    )
    _, ax = plt.subplots(figsize=(9, 6))
    ax.bar(bin_centers, hist, width=bin_width, alpha=0.5, yerr=std_dev)
    ax.set_xlabel("Likelihood value")
    ax.set_ylabel("#Events (normalized)")

    if savefig:
        plt.savefig("likelihood_top.pdf", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    # data_higgs = get_data(
    #     parquet_file_path="/Users/quint/Documents/Studium/Hiwi_Master/files/columns_0_higgs.parquet"
    # )
    bin_nr = 40
    data_top = get_data(parquet_file_path="/Users/quint/Documents/Studium/Hiwi_Master/files/columns_0_top.parquet")

    # counts_higgs, errors_higgs = higgs_likelihood(data_higgs, bin_nr)
    counts_top, errors_top = top_likelihood(data_top, bin_nr)
    plot_likelihood(counts_top, errors_top, savefig=True)
