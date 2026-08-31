from hbt.production.histogram_helper_functions import create_signal_hist, create_background_hist
from columnflow.util import maybe_import

np = maybe_import("numpy")

if __name__ == "__main__":
    parquet_file_path_signal = "/data/dust/user/diepholq/hh2bbtautau/hist_input_data/higgs/columns_0.parquet"
    parquet_file_path_bg = "/data/dust/user/diepholq/hh2bbtautau/hist_input_data/top/"
    print("Creating signal hist...")
    create_signal_hist(
        parquet_file_path_signal,
        1480,
        np.array([38, 38]),
    )
    print("Done.")
    print("Creating background hist...")
    create_background_hist(
        parquet_file_path_bg,
        1480,
        np.array([21, 8, 8]),
    )
    print("Done.")
