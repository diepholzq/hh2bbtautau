from hbt.production.histogram_helper_functions import create_signal_hist, create_background_hist
from columnflow.util import maybe_import
np = maybe_import("numpy")

if __name__ == "__main__":
    parquet_file_path_signal = (
        "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/"
        "hh_ggf_hbb_htt_kl1_kt1_powheg/nominal/calib__default/sel__default/red__default/prod__pdf_inputs/"
        "dev_likelihood_ratio/columns_0.parquet"
    )
    parquet_file_path_bg = (
        "/data/dust/user/diepholq/hh2bbtautau/hbt_store/analysis_hbt/cf.ProduceColumns/22pre_v14/tt_dl_powheg/"
        "nominal/calib__default/sel__default/red__default/prod__pdf_inputs/dev_likelihood_ratio/"
    )
    create_signal_hist(
        parquet_file_path_signal,
        750,
        np.array([27, 27]),
    )
    create_background_hist(
        parquet_file_path_bg,
        750,
        np.array([20, 7, 7]),
    )
