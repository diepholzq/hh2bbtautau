from columnflow.util import maybe_import
import pickle
from sympy.utilities.iterables import variations
from glob import glob

ak = maybe_import("awkward")
np = maybe_import("numpy")


def multiply_string_arr(names_arr: np.array) -> str:
    return_str = ""
    for idx, name in enumerate(names_arr):
        if idx != 0:
            return_str = f"{return_str} * {name}"
        else:
            return_str = f"{name}"
    return return_str


def error_formula(value_names: np.array, uncert_names: np.array) -> str:
    """Returns a string that can be evaluated to calculate the statistical uncertainties.
    Args:
        value_names (np.array): Array with names of the variables where the likelihood values per fact. step are stored
        uncert_names (np.array): Array with names of the variables where the stat. uncert. per fact. step are stored
    """
    num_fact_steps: int = len(value_names)
    formula_string = "np.sqrt("
    idx_arr = np.arange(0, num_fact_steps, 1)
    for idx, fact_step in enumerate(value_names):
        uncert_arr_step: np.array = uncert_names[idx_arr[idx_arr != idx]]
        uncert_str = multiply_string_arr(uncert_arr_step)
        if idx != idx_arr[-1]:
            to_append = f"({fact_step} * {uncert_str})**2 + "
        else:
            to_append = f"({fact_step} * {uncert_str})**2)"
        formula_string = f"{formula_string}{to_append}"
    return formula_string


def roc_auc_std(A: float, n_p: float, n_n: float) -> str:
    """Calculates the statistical error on the roc auc score
    Args:
        A (float): auc score
        n_p (float): number of positive cases in sample
        n_n (float): number of negative cases in sample
    Returns:
        roc_auc_std (float): standard error of auc score
    """
    D_p = (n_p - 1) * (A / (2 - A) - A**2)
    D_n = (n_n - 1) * ((2 * A**2) / (1 + A) - A**2)
    roc_auc_std = np.sqrt((A * (1 - A) + D_p + D_n) / (n_p * n_n))
    return roc_auc_std


def get_data(
        parquet_file_path: str,
        column_name: str,
        drop_nones: bool = False,
) -> ak.Array:
    """Extracts the relevant fields from input data (meaning signal genlvl MC)
    Args:
        parquet_file_path (str): Path where the parquet input file is located
        column_name (str): Name of the column containing the inputs
        drop_nones (bool): If nones present in data should be removed
    Returns:
        data (ak.Array): An ak.Array containing the relevant fields
    """
    objects = ak.from_parquet(parquet_file_path)
    objects = objects
    data = eval(f"objects.{column_name}")
    if drop_nones:
        data = ak.drop_none(data, axis=0)
    return data


def flat_binning(x: np.array, n_bins: int):
    """Computes bin edges to histogram x, where there is the same amount of entries in each bin
    Args:
        x (np.array): Data to be binned
        NBins (int): Number of bins
    Returns:
        edges (np.array): Bin edges
        entries (np.array): Entries per bin
    """
    percentile_divider = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(x, percentile_divider)
    entries, _ = np.histogram(x, bins=edges)
    return edges, entries


def create_multidim_hist(
    data: ak.Array,
    field_list: np.ndarray,
    bins_per_dim: np.ndarray,
) -> dict:
    """Build a multidimensional (2 or 3 dim.) histogram from the input data and the fields provided, with a given number
    of bins per dimension. The histogram is normalized so that the n-d area integrates to 1. The binning is such that
    there is an (almost) equal amount of statistics in each bin.
    Args:
        data (ak.Array): The data to fill the histogram
        field_list (numpy.ndarray): A list containing the field names which will make up the dimensions of the
                                    histogram as strings
        bins_per_dim (np.ndarray): A list containing the number of bins per dimension. Dimensions will be assigned
                                   according to field_list, i.e. bins_per_dim[0] will assign number of bins together
                                   field_list[0]
    Returns:
        hist_returns (dict): Dictionary containing the nd-'histogram' and a list of np.ndarray containing the edges for
        each dimension, together with an nd-histogram with the same binning, containing the statistical uncerainties per
        bin
    """
    # Get data
    n_dims = len(field_list)
    if (n_dims < 2) or (n_dims > 3):
        raise Exception("Wrong number of dimensions. n_dims must be 2 or 3")
    data_x = ak.to_numpy(data[field_list[0]])
    data_y = ak.to_numpy(data[field_list[1]])
    if n_dims == 3:
        data_z = ak.to_numpy(data[field_list[2]])

    # Create empty histogram and edges (entries hist for stat. uncert. estimation)
    edges_dim1 = np.zeros(bins_per_dim[0] + 1)
    edges_dim2 = np.zeros((bins_per_dim[0], bins_per_dim[1] + 1))
    if n_dims == 3:
        pdf = np.zeros((bins_per_dim[0], bins_per_dim[1], bins_per_dim[2]))
        edges_dim3 = np.zeros((bins_per_dim[0], bins_per_dim[1], bins_per_dim[2] + 1))
    else:
        pdf = np.zeros((bins_per_dim[0], bins_per_dim[1]))

    entries = np.zeros_like(pdf)
    # Calculate statistical binning in first dimension
    edges1, entries1 = flat_binning(data_x, bins_per_dim[0])
    d_edges1 = np.diff(edges1)
    edges_dim1[:] = edges1

    # Loop through bins of first dimension: For each bin, select all events from other dimensions (y and possibly z)
    # with x-values that fall into that bin. Go iteratively through these dimensions and caclulate the statistical
    # binning.
    # This will result in a different binning choice in dimension y for each bin in dimension x (+ the same for z and y)
    for idx1 in range(bins_per_dim[0]):
        # get bin edges:
        left_edge = edges1[idx1]
        right_edge = edges1[idx1 + 1]
        # select entries in other dims that fall into this bin
        entries_dim_2 = data_y[(data_x >= left_edge) * (data_x < right_edge)]

        # Calculate stat. bin edges for this x-bin
        edges2, entries2 = flat_binning(entries_dim_2, bins_per_dim[1])
        edges_dim2[idx1, :] = edges2
        # Calculate bin widths for bin volume calculation
        d_edges2 = np.diff(edges2)

        # print('idx1, edges2(min,max)', idx1, edges2[0], edges2[-1])
        # plt.hist(entries_dim_2, bins=edges2, histtype='step')

        if n_dims == 3:
            entries_dim_3 = data_z[(data_x >= left_edge) * (data_x < right_edge)]
            for idx2 in range(bins_per_dim[1]):
                # Same logic as above
                left_edge = edges2[idx2]
                right_edge = edges2[idx2 + 1]
                # Select entries in dim 3 that fall into current dim 2 bin
                entries_from_3_in_2 = entries_dim_3[(entries_dim_2 >= left_edge) * (entries_dim_2 < right_edge)]
                edges3, entries3 = flat_binning(entries_from_3_in_2, bins_per_dim[2])
                edges_dim3[idx1, idx2, :] = edges3
                d_edges3 = np.diff(edges3)

                # Calculate bin volume for each bin, fill pdf and entries histograms
                for idx3 in range(bins_per_dim[2]):
                    volume = d_edges1[idx1] * d_edges2[idx2] * d_edges3[idx3]
                    pdf[idx1, idx2, idx3] = entries3[idx3] / len(data_y) / volume
                    entries[idx1, idx2, idx3] = entries3[idx3]

                # plt.hist(entries_from_3_in_2, bins=edges3, histtype='step')
        else:
            for idx2 in range(bins_per_dim[1]):
                # Same logic as above
                volume = d_edges1[idx1] * d_edges2[idx2]
                pdf[idx1, idx2] = entries2[idx2] / len(data_y) / volume
                entries[idx1, idx2] = entries2[idx2]

    edges_dict = {
        "edges_dim1": edges_dim1,
        "edges_dim2": edges_dim2,
    }
    if n_dims == 3:
        edges_dict["edges_dim3"] = edges_dim3

    norm_factor_hist = np.nan_to_num(pdf / entries, posinf=0.0, neginf=0.0)   # bin contents: 1/(n_events * bin_volume)
    uncertainty_hist = abs(norm_factor_hist * np.sqrt(entries))

    hist_returns: dict = {
        "hist": pdf,
        "edges": edges_dict,
        "uncertainty_hist": uncertainty_hist,
    }
    # plt.show()
    # plt.close()
    return hist_returns


def create_any_hist(
    data: ak.Array,
    field_list: np.ndarray,
    bins: int = 10,
    statistical_binning: bool = True,
    bin_filling: bool = False,
):
    """Build a multidimensional histogram from the input data and the fields provided, with a given number of bins per
    dimension. The histogram is normalized so that the n-d area integrates to 1. Statistical / percentile binning only
    works in 1D. So, the name of the function is a bit misleading.
    Args:
        data (ak.Array): The data to fill the histogram
        field_list (numpy.ndarray): A list containing the field names which will make up the dimensions of the
        histogram as strings
        bins (int): The number of bins per dimension
        statistical_binning (bool): If binning should be used that ensures defined amount of statistics per bin (define
        via percentile_divider)
        bin_filling (bool): If empty bins should be filled by mean of neighbouring values
    Returns:
        hist_returns (dict): Dictionary containing the nd-histogram and a list of np.ndarray containing the edges for
        each dimension, together with an nd-histogram with the same binning, containing the statistical uncerainties per
        bin, and the number of empty bins
    """
    # Build coordinate_list as input for hist
    n_dims: int = len(field_list)
    coordinate_list = np.zeros((len(data), n_dims))
    bin_list = np.empty((bins + 1, n_dims))
    # if bins == 4:
    #     divider = 100 / bins
    #     percentile_divider = np.arange(0, 100 + divider, divider)
    # else:
    percentile_divider = np.concatenate([[0], np.linspace(5, 95, bins - 1), [100]], axis=0)
    idx = 0
    for field in field_list:
        coordinate_list[:, idx] = data[field]
        bin_list[:, idx] = np.percentile(coordinate_list[:, idx], percentile_divider)
        idx += 1
    bin_list = bin_list.T
    # Define bins per dimension

    # Create hist
    # Each histogram is normalized so that the n-dim. integral = 1
    # bin_contents = = N_i/N/V_i:
    if statistical_binning:
        hist, edges = np.histogramdd(coordinate_list, bins=bin_list, density=True)
    else:
        print("\nNot using statistical binning\n")
        hist, edges = np.histogramdd(coordinate_list, bins=bins, density=True)

    # fill empty bins with average of neighbouring values
    # step 1: find empty bins and store their coordinates
    if bin_filling:
        empty_mask = hist == 0
        tr = True
        n_empty: int = len(ak.ravel(empty_mask)[ak.ravel(empty_mask) == tr])
        empty_coordinates = np.zeros((n_empty, n_dims), dtype=int)

        nul_counter: int = 0
        it = np.nditer(hist, flags=["multi_index"])
        for entry in it:
            if entry == 0:
                empty_coordinates[nul_counter] = it.multi_index
                nul_counter += 1
        # step 2: get coordinates of neighbouring values
        num_neighbours = 2**n_dims
        variation_arr = np.zeros((num_neighbours, n_dims), dtype=int)
        for nb_idx, nb in enumerate(variations([+1, -1], n_dims, True)):
            for dim_idx, dim_var in enumerate(nb):
                variation_arr[nb_idx, dim_idx] = dim_var
                # if n_dims > 1:
                #     variation_arr[nb_idx, dim_idx] = dim_var
                # else:
                #     variation_arr[nb_idx] = dim_var
        neighbour_coords = np.reshape(
            np.repeat(empty_coordinates, num_neighbours, axis=0), (n_empty, num_neighbours, n_dims)
        )
        variation_arr = np.resize(variation_arr, (n_empty, num_neighbours, n_dims))
        neighbour_coords = neighbour_coords + variation_arr
        # make sure coordinates stay in allowed range
        neighbour_coords[neighbour_coords == bins] = bins - 1
        neighbour_coords[neighbour_coords < 0] = 0
        # step 3: get values of neighbours and compute average
        for empty_bin, coord in enumerate(empty_coordinates):
            neighbour_avg = np.mean(hist[tuple(neighbour_coords[empty_bin].T)])
            hist[tuple(coord)] = neighbour_avg

    # Calculate statistical uncerainties per bin
    if statistical_binning:
        N_i, _ = np.histogramdd(coordinate_list, bins=bin_list, density=False)         # bin_contents = N_i
    else:
        N_i, _ = np.histogramdd(coordinate_list, bins=bins, density=False)         # bin_contents = N_i
        print("\nNot using statistical binning\n")

    if bin_filling:
        for empty_bin, coord in enumerate(empty_coordinates):
            neighbour_avg = np.mean(hist[tuple(neighbour_coords[empty_bin].T)])
            N_i[tuple(coord)] = neighbour_avg
    norm_factor_hist = np.nan_to_num(hist / N_i, nan=0, posinf=0)                  # bin_contents = 1/(N*V_i)
    uncertainty_hist = np.sqrt((norm_factor_hist * np.sqrt(N_i))**2)

    # Count emtpy bins:
    empty_mask = N_i == 0
    tr = True
    n_empty: int = len(ak.ravel(empty_mask)[ak.ravel(empty_mask) == tr])
    print(f"\n\nNumber of bins for field list {field_list}: {bins**n_dims}")
    print(f"{n_empty} of those are empty, i.e. {n_empty / (bins**n_dims) * 100}%\n\n")
    edges = {"edges_dim1": edges[0]}
    hist_returns: dict = {
        "hist": hist,
        "edges": edges,
        "uncertainty_hist": uncertainty_hist,
        "n_empty": n_empty,
    }
    return hist_returns


def get_event_likelihood_1d(
    events: ak.Array,
    hist: np.ndarray,
    edges: list,
    uncertainty_hist: np.ndarray,
    fields: np.ndarray,
    bin_nr: int = 10,
) -> np.ndarray:
    """Evaluates the likelihood (hist) on the provided events. Used for evaluating one factorization step
    Args:
        events (ak.Array): The data the likelihood is to be evaluated on
        hist (np.ndarray): The multidimensional histogram
        edges (list): The list of np.ndarrays containing the bin edges along each dimension
        uncertainty_hist (np.ndarray): The multidimensional histogram of the stat. err.
        fields (list): The fields of data
        bin_nr (int): Number of bins per dimension
    Returns:
        Tuple(np.ndarray, np.ndarray): The evaluated histogram, i.e. the likelihood score for each event,
        together with event-wise uncerainties in a second array
    """

    # This would be a normalization of the eval output per factorization step
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
        bin_indices[:, idx] = np.digitize(events[field],
                                          edges[idx][1:],
                                          right=True)  # type: ignore
        idx += 1
    bin_indices[bin_indices == bin_nr] = bin_nr - 1
    bin_indices = bin_indices.astype(dtype=np.int32)
    bin_indices = bin_indices.T

    # return bin values for each event, they are a measure of the prob. of that event being signal-like at that step
    return hist[tuple(bin_indices)], uncertainty_hist[tuple(bin_indices)]


def get_event_likelihood_nd(
    data: ak.Array,
    hist: np.ndarray,
    edges_dict: dict,
    uncertainty_hist: np.ndarray,
    field_list: np.ndarray,
    bins_per_dim: np.ndarray,
):
    """
    """
    n_dims = len(field_list)
    data_dim1 = ak.to_numpy(data[field_list[0]])
    bin_indices_dim1 = np.zeros(len(data[field_list[0]]), dtype=np.int32)
    if n_dims > 1:
        data_dim2 = ak.to_numpy(data[field_list[1]])
        bin_indices_dim2 = np.zeros(len(data[field_list[1]]), dtype=np.int32)
    if n_dims == 3:
        data_dim3 = ak.to_numpy(data[field_list[2]])
        bin_indices_dim3 = np.zeros(len(data[field_list[2]]), dtype=np.int32)
    bin_indices_dim1 = np.digitize(data_dim1, edges_dict["edges_dim1"], right=True) - 1
    bin_indices_dim1[bin_indices_dim1 == -1] = 0
    bin_indices_dim1[bin_indices_dim1 > bins_per_dim[0] - 1] = bins_per_dim[0] - 1

    if n_dims > 1:
        print(f"Evaluating... fields: {field_list}")
        for ev in range(len(data_dim2)):
            bin_index = np.digitize(
                data_dim2[ev],
                edges_dict["edges_dim2"][bin_indices_dim1[ev]],
                right=True,
            ) - 1
            bin_index = min(bin_index, bins_per_dim[1] - 1)
            bin_index = max(bin_index, 0)

            bin_indices_dim2[ev] = bin_index
            if n_dims == 3:
                from IPython import embed
                bin_index_dim3 = np.digitize(
                    data_dim3[ev],
                    edges_dict["edges_dim3"][bin_indices_dim1[ev],
                    bin_indices_dim2[ev]],
                    right=True,
                )
                bin_index_dim3 = min(bin_index_dim3, bins_per_dim[2] - 1)
                bin_index_dim3 = max(bin_index_dim3, 0)
                bin_indices_dim3[ev] = bin_index_dim3
            if ev % 10000 == 0:
                print(f"@ event {ev} of {len(data_dim2)}", end="\r")
    print("\n")

    # Handle overflow and underflow behaviour -> Put in last and first bin
    # if n_dims > 2:
    #     ind_lis = [bin_indices_dim1, bin_indices_dim2, bin_indices_dim3]
    # elif n_dims == 2:
    #     ind_lis = [bin_indices_dim1, bin_indices_dim2]
    # elif n_dims == 1:
    #     ind_lis = [bin_indices_dim1]
    # else:
    #     raise Exception("Wrong number of dimensions")
    # for lis in ind_lis:
    #     lis[lis == -1] = 0
    #     lis[lis > n_bins - 1] = n_bins - 1
    if n_dims == 3:
        bin_indices = np.concatenate([
            bin_indices_dim1[:, None],
            bin_indices_dim2[:, None],
            bin_indices_dim3[:, None],
        ], axis=1,
        )
    elif n_dims == 2:
        bin_indices = np.concatenate([
            bin_indices_dim1[:, None],
            bin_indices_dim2[:, None],
        ], axis=1,
        )
    elif n_dims == 1:
        bin_indices = np.concatenate([
            bin_indices_dim1[:, None],
        ], axis=1,
        )
    else:
        raise Exception("Wrong number of dimensions")
    # Evaluate hist:
    try:
        evaluated = hist[tuple(bin_indices.T)]
    except:
        embed()
    stat_uncerts = uncertainty_hist[tuple(bin_indices.T)]

    return evaluated, stat_uncerts


def create_top_likelihood_hists(
    top_data: ak.Array,
    bins_per_dim: np.ndarray = np.array([15, 7, 7]),
    bin_nr: int = 30,
    hist_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/",
    statistical_binning: bool = True,
    bin_filling: bool = False,
):
    """Fills the factorization step histograms with events from the desired dataset. Pickles the histograms as Dictionary
    so they can be loaded for evaluation. Naming convention: top_hists_nbins{bin_nr}.pkl
    Stored in hist_path.
    Args:
        top_data (ak.Array): Array containing the column with the top likelihood inputs
        bin_nr (int): Number of bins per 1d factorization step
        bins_per_dim (np.ndarray): A list containing the number of bins per dimension of the multidim. fact. step
        statistical_binning (bool): If the statistical_binning tweak should be used
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
    """
    data = top_data
    step1_dict = create_multidim_hist(
        data,
        field_list=np.array(
            ["tt_vis_system_mass", "tau1_cos_theta_star_cms_wplus", "tau2_cos_theta_star_cms_wminus"],
        ),
        bins_per_dim=bins_per_dim,
    )
    step2_dict = create_any_hist(
        data,
        field_list=np.array(["tt_vis_system_pt"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling,
    )

    step3_dict = create_any_hist(
        data,
        field_list=np.array(["tt_vis_system_pz"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling,
    )

    step4_dict = create_any_hist(
        data,
        field_list=np.array(["t_vis_y_diff"]),
        bins=bin_nr,
    )

    step5_dict = create_any_hist(
        data,
        field_list=np.array(["tau1_cos_theta_star_cms_t1_vis"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling,
    )

    step6_dict = create_any_hist(
        data,
        field_list=np.array(["tau2_cos_theta_star_cms_t2_vis"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling,
    )
    hist_dict = {
        "step1_dict": step1_dict,
        "step2_dict": step2_dict,
        "step3_dict": step3_dict,
        "step4_dict": step4_dict,
        "step5_dict": step5_dict,
        "step6_dict": step6_dict,
    }
    with open(
        f"{hist_path}top_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl", "wb",
    ) as hist_file:
        pickle.dump(hist_dict, hist_file)


def create_higgs_likelihood_hists(
    higgs_data: ak.Array,
    hist_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/",
    bin_nr: int = 30,
    bins_per_dim: np.ndarray = np.array([25, 25]),
    statistical_binning: bool = True,
    bin_filling: bool = False,
):
    """Fills the signal factorization step histograms with events from the desired dataset. Pickles the histograms as
    dictionary so they can be loaded for evaluation. Naming convention: higgs_hists_nbins{bin_nr}.pkl
    Stored in hist_path.
    Args:
        higgs_data (ak.Array): Array containing the column with the top likelihood inputs
        bin_nr (int): Number of bins for each n-d hist
        bins_per_dim (np.ndarray): A list containing the number of bins per dimension of the multidim. fact. step
        statistical_binning (bool): If the statistical_binning tweak should be used
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
    Returns:
        Tuple(np.ndarry, np.ndarray): Array with the event probabilities, Array with the statistical uncertainties per
        event
    """
    data = higgs_data

    step1_dict = create_multidim_hist(data,
        np.array([
            "dihiggs_mass",
            "dihiggs_system_pt",
        ]), bins_per_dim=bins_per_dim)

    step2_dict = create_any_hist(data,
        field_list=np.array([
            "dihiggs_system_pz",
        ]), bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling)

    step3_dict = create_any_hist(data,
        field_list=np.array(["cos_theta_cms_h1_b1"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling)

    step4_dict = create_any_hist(data,
        field_list=np.array(["cos_theta_h1"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling)

    step5_dict = create_any_hist(data,
        field_list=np.array(["cos_theta_cms_h2_tau_vis1"]),
        bins=bin_nr, statistical_binning=statistical_binning, bin_filling=bin_filling)

    hist_dict = {
        "step1_dict": step1_dict,
        "step2_dict": step2_dict,
        "step3_dict": step3_dict,
        "step4_dict": step4_dict,
        "step5_dict": step5_dict,
    }
    with open(
        f"{hist_path}higgs_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl", "wb",
    ) as hist_file:
        pickle.dump(hist_dict, hist_file)


def create_signal_hist(
    parquet_file_path_signal: str,
    bin_nr: str,
    bins_per_dim: str,
    hist_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/",
    statistical_binning: bool = True,
) -> ak.Array:
    """Creates likelihood histograms for the signal likelihood with the desired options.
    Options need to be set manually below atm.
    """
    # Check if hist already exists
    hist_file_path: str = f"{hist_path}higgs_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl"
    try:
        with open(hist_file_path, "rb") as hist_file:
            _ = pickle.load(hist_file)
    except:
        # Get signal events and input column:
        parquet_file_path_signal = (
            "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
            "hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/"
            "dev_likelihood_ratio/columns_0.parquet"
        )
        signal_data = get_data(parquet_file_path_signal, "pdf_input_vars_reco_higgs", drop_nones=True)

        # Create hist
        create_higgs_likelihood_hists(
            signal_data,
            hist_path=hist_path,
            bin_nr=bin_nr,
            bins_per_dim=bins_per_dim,
            statistical_binning=statistical_binning,
            bin_filling=False,
        )
    else:
        print(f"Histogram {hist_file_path} already exists...")


def create_background_hist(
    parquet_file_path_signal: str,
    bin_nr: str,
    bins_per_dim: str,
    hist_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/",
    statistical_binning: bool = True,
) -> ak.Array:
    """Creates likelihood histograms for the background likelihood with the desired options.
    Options need to be set manually below atm.
    """
    # Check if hist already exists
    hist_file_path: str = f"{hist_path}top_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl"
    parquet_file_path_bg = (
        "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/tt_dl_powheg/"
        "nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"
    )
    try:
        with open(hist_file_path, "rb") as hist_file:
            _ = pickle.load(hist_file)
    except:
        # Get background events and input column:
        # Check if individual files are already added together:
        try:
            bg_data = get_data(
                f"{parquet_file_path_bg}columns_all.parquet", "pdf_input_vars_reco_top", drop_nones=True,
            )
        except:
            file_list = glob(f"{parquet_file_path_bg}*.parquet")
            result = ak.concatenate(
                [ak.from_parquet(file_list[0]),
                ak.from_parquet(file_list[1])], axis=0)
            for idx in range(2, len(file_list)):
                result = ak.concatenate([result, ak.from_parquet(file_list[idx])], axis=0)
                ak.to_parquet(result, f"{parquet_file_path_bg}columns_all.parquet")

            bg_data = get_data(
                f"{parquet_file_path_bg}columns_all.parquet", "pdf_input_vars_reco_top", drop_nones=True,
            )

        # Create hist
        create_top_likelihood_hists(
            bg_data,
            hist_path=hist_path,
            bin_nr=bin_nr,
            bins_per_dim=bins_per_dim,
            statistical_binning=statistical_binning,
            bin_filling=False,
        )
    else:
        print(f"Histogram {hist_file_path} already exists...")


def eval_top_likelihood_hists(
    eval_data: ak.Array,
    bin_nr: int,
    allow_hist_creation: bool = False,
    bins_per_dim: np.array = np.array([15, 7, 7]),
    statistical_binning: bool = True,
    bin_filling: bool = False,
    do_constr: bool = True,
    hist_file_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/top_hists.pkl",
):
    """Evaluates given data on the top likelihood histograms and returns L_hists
    Args:
        hist_file_path (str): Path to the pickle file containting the filled histogram dictionary
        bins_per_dim (np.ndarray): A list containing the number of bins per dimension of the multidim. fact. step
    """
    hist_file_path = (
        f"/data/dust/user/diepholq/hh2bbtautau/hist_files/"
        f"top_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl"
    )
    try:
        with open(hist_file_path, "rb") as hist_file:
            hist_dict = pickle.load(hist_file)
    except:
        if not allow_hist_creation:
            raise Exception(f"Histogram {hist_file_path} not found, please create first")
        else:
            parquet_file_path_bg = (
                "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/tt_dl_powheg/"
                "nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"
            )
            create_background_hist(
                parquet_file_path_bg,
                bin_nr,
                bins_per_dim,
            )
        with open(hist_file_path, "rb") as hist_file:
            hist_dict = pickle.load(hist_file)
    step1_dict = hist_dict["step1_dict"]
    step2_dict = hist_dict["step2_dict"]
    step3_dict = hist_dict["step3_dict"]
    step4_dict = hist_dict["step4_dict"]
    step5_dict = hist_dict["step5_dict"]
    step6_dict = hist_dict["step6_dict"]
    hist_step1, edges_step1, uncert_hist_step1 = step1_dict["hist"], step1_dict["edges"], step1_dict["uncertainty_hist"]
    hist_step2, edges_step2, uncert_hist_step2 = step2_dict["hist"], step2_dict["edges"], step2_dict["uncertainty_hist"]
    hist_step3, edges_step3, uncert_hist_step3 = step3_dict["hist"], step3_dict["edges"], step3_dict["uncertainty_hist"]
    hist_step4, edges_step4, uncert_hist_step4 = step4_dict["hist"], step4_dict["edges"], step4_dict["uncertainty_hist"]
    hist_step5, edges_step5, uncert_hist_step5 = step5_dict["hist"], step5_dict["edges"], step5_dict["uncertainty_hist"]
    hist_step6, edges_step6, uncert_hist_step6 = step6_dict["hist"], step6_dict["edges"], step6_dict["uncertainty_hist"]
    p1, errors1 = get_event_likelihood_nd(
        eval_data,
        hist_step1,
        edges_step1,
        uncert_hist_step1,
        np.array(["tt_vis_system_mass", "tau1_cos_theta_star_cms_wplus", "tau2_cos_theta_star_cms_wminus"]),
        bins_per_dim=bins_per_dim,
    )
    bin_nr = np.array([bin_nr])
    p2, errors2 = get_event_likelihood_nd(
        eval_data,
        hist_step2,
        edges_step2,
        uncert_hist_step2,
        np.array(["tt_vis_system_pt"]),
        bin_nr,
    )

    p3, errors3 = get_event_likelihood_nd(
        eval_data,
        hist_step3,
        edges_step3,
        uncert_hist_step3,
        np.array(["tt_vis_system_pz"]),
        bin_nr,
    )

    p4, errors4 = get_event_likelihood_nd(
        eval_data,
        hist_step4,
        edges_step4,
        uncert_hist_step4,
        np.array(["t_vis_y_diff"]),
        bin_nr,
    )

    p5, errors5 = get_event_likelihood_nd(
        eval_data,
        hist_step5,
        edges_step5,
        uncert_hist_step5,
        np.array(["tau1_cos_theta_star_cms_t1_vis"]),
        bin_nr,
    )

    p6, errors6 = get_event_likelihood_nd(
        eval_data,
        hist_step6,
        edges_step6,
        uncert_hist_step6,
        np.array(["tau2_cos_theta_star_cms_t2_vis"]),
        bin_nr,
    )
    value_names = np.array(["p1", "p2", "p3", "p4", "p5", "p6"])
    uncert_names = np.array(["errors1", "errors2", "errors3", "errors4", "errors5", "errors6"])
    p_top = eval(multiply_string_arr(value_names)) * 2 / np.pi
    errors_p_top = error_formula(value_names, uncert_names)
    errors_p_top = eval(errors_p_top)
    results_dict = {
        "p_top": p_top,
        "errors_p_top": errors_p_top,
    }
    return results_dict


def eval_higgs_likelihood_hists(
    eval_data: ak.Array,
    bin_nr: int,
    allow_hist_creation: bool = False,
    bins_per_dim: np.ndarray = np.array([25, 25]),
    statistical_binning: bool = True,
    bin_filling: bool = False,
    do_constr: bool = True,
    hist_file_path: str = "/data/dust/user/diepholq/hh2bbtautau/hist_files/higgs_hists.pkl",
):
    """Evaluates given data on the higgs likelihood histograms and return L_hists
        eval_data (ak.Array): Data to evaluate the likelihood on
        bin_nr (int): Number of bins for each n-d hist
        bins_per_dim (np.ndarray): A list containing the number of bins per dimension of the multidim. fact. step
        statistical_binning (bool): If the statistical_binning tweak should be used
        bin_filling (bool): If empty bins should be filled w/ mean of neighb. vals
        hist_file_path (str): Path to the pickle file containting the filled histogram dictionary
    """
    hist_file_path = (
        f"/data/dust/user/diepholq/hh2bbtautau/hist_files/"
        f"higgs_hists_bpd{bins_per_dim}_nbins{bin_nr}.pkl"
    )
    try:
        with open(hist_file_path, "rb") as hist_file:
            hist_dict = pickle.load(hist_file)
    except:
        if not allow_hist_creation:
            raise Exception(f"Histogram {hist_file_path} not found, please create first")
        else:
            parquet_file_path_signal = (
                "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
                "hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/"
                "dev_likelihood_ratio/columns_0.parquet"
            )
            create_signal_hist(
                parquet_file_path_signal,
                bin_nr,
                bins_per_dim,
            )
        with open(hist_file_path, "rb") as hist_file:
            hist_dict = pickle.load(hist_file)
    step1_dict = hist_dict["step1_dict"]
    step2_dict = hist_dict["step2_dict"]
    step3_dict = hist_dict["step3_dict"]
    step4_dict = hist_dict["step4_dict"]
    step5_dict = hist_dict["step5_dict"]
    hist_step1, edges_step1, uncert_hist_step1 = step1_dict["hist"], step1_dict["edges"], step1_dict["uncertainty_hist"]
    hist_step2, edges_step2, uncert_hist_step2 = step2_dict["hist"], step2_dict["edges"], step2_dict["uncertainty_hist"]
    hist_step3, edges_step3, uncert_hist_step3 = step3_dict["hist"], step3_dict["edges"], step3_dict["uncertainty_hist"]
    hist_step4, edges_step4, uncert_hist_step4 = step4_dict["hist"], step4_dict["edges"], step4_dict["uncertainty_hist"]
    hist_step5, edges_step5, uncert_hist_step5 = step5_dict["hist"], step5_dict["edges"], step5_dict["uncertainty_hist"]

    # Get "probabilities" / evaluate eval_data
    probs_step1, errors_step1 = get_event_likelihood_nd(
        eval_data,
        hist_step1,
        edges_step1,
        uncert_hist_step1,
        np.array([
            "dihiggs_mass",
            "dihiggs_system_pt",
        ]),
        bins_per_dim,
    )
    bin_nr = np.array([bin_nr])
    probs_step2, errors_step2 = get_event_likelihood_nd(
        eval_data,
        hist_step2,
        edges_step2,
        uncert_hist_step2,
        np.array([
            "dihiggs_system_pz",
        ]),
        bin_nr,
    )

    probs_step3, errors_step3 = get_event_likelihood_nd(
        eval_data,
        hist_step3,
        edges_step3,
        uncert_hist_step3,
        np.array(["cos_theta_h1"]),
        bin_nr,
    )

    probs_step4, errors_step4 = get_event_likelihood_nd(
        eval_data,
        hist_step4,
        edges_step4,
        uncert_hist_step4,
        np.array(["cos_theta_cms_h1_b1"]),
        bin_nr,
    )

    probs_step5, errors_step5 = get_event_likelihood_nd(
        eval_data,
        hist_step5,
        edges_step5,
        uncert_hist_step5,
        np.array(["cos_theta_cms_h2_tau_vis1"]),
        bin_nr,
    )

    # Calculate statistical uncertainties for each fact. step
    # Calculate final Likelihood score for each event
    value_names = np.array(
        ["probs_step1", "probs_step2", "probs_step3", "probs_step4", "probs_step5"],
    )
    uncert_names = np.array(
        ["errors_step1", "errors_step2", "errors_step3", "errors_step4", "errors_step5"],
    )
    probs_higgs = eval(multiply_string_arr(value_names)) * 2 / np.pi

    # Uncertainty propagation (Gaussche Fehlerfortpflanzung)
    errors_higgs = eval(error_formula(value_names, uncert_names))

    results_dict = {
        "probs_higgs": probs_higgs,
        "errors_higgs": errors_higgs,
    }
    return results_dict
