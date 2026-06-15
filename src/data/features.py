import os
import pathlib


def expand(str_list):
    def brace_expand(s):
        # "a{b,c}d" -> ["abd", "acd"]
        if "{" not in s:
            return [s]
        pre, post = s.split("{", 1)
        mid, post = post.split("}", 1)
        parts = mid.split(",")
        expanded = []
        for part in parts:
            for rest in brace_expand(post):
                expanded.append(pre + part + rest)
        return expanded

    cols = []
    for _str in str_list:
        cols.extend(brace_expand(_str))
    return cols


def add_prefix(string, prefix, ignore_code="_"):
    if string.startswith(ignore_code):
        return string[1:]  # remove ignoring code
    else:
        return prefix + string


def prefix_map():
    stem = pathlib.Path(os.environ["INPUT_DATA_DIR"]).stem
    stem_to_prefix = {
        "prod14": "res_dnn_pnet",
        "prod20_vbf": "reg_dnn_moe",
        "prod20": "reg_dnn_moe",
        "prod19": "res_dnn_pnet",
        "prod24": "reg_dnn_moe",
    }
    return stem_to_prefix[stem]


def expected_embedding_inputs():
    data_prefix = prefix_map()
    embedding_expected_inputs = {
        f"{data_prefix}_pair_type": [0, 1, 2],  # see mapping below
        f"{data_prefix}_dm1": [-1, 0, 1, 10, 11],  # -1 for e/mu
        f"{data_prefix}_dm2": [0, 1, 10, 11],
        f"{data_prefix}_vis_tau1_charge": [-1, 1],
        f"{data_prefix}_vis_tau2_charge": [-1, 1],
        f"{data_prefix}_has_fatjet": [0, 1],  # whether a selected fatjet is present
        f"{data_prefix}_has_jet_pair": [0, 1],  # whether two or more jets are present
        # 0: 2016APV, 1: 2016, 2: 2017, 3: 2018, 4: 2022preEE, 5: 2022postEE, 6: 2023pre, 7: 2023post
        "{data_prefix}_year_flag": [0, 1, 2, 3, 4, 5, 6, 7],
        "channel_id": [1, 2, 3],
    }
    return embedding_expected_inputs


def input_features(debug=False, debug_length=3):
    data_prefix = prefix_map()
    categorical_features: list[str] = [
        "pair_type",
        # "_channel_id",
        "dm1",
        "dm2",
        "vis_tau1_charge",
        "vis_tau2_charge",
        "has_jet_pair",
        "has_fatjet",
    ]

    continous_features: list[str] = [
        "met_px",
        "met_py",
        "met_cov00",
        "met_cov01",
        "met_cov11",
        "vis_tau1_px",
        "vis_tau1_py",
        "vis_tau1_pz",
        "vis_tau1_e",
        "vis_tau2_px",
        "vis_tau2_py",
        "vis_tau2_pz",
        "vis_tau2_e",
        "bjet1_px",
        "bjet1_py",
        "bjet1_pz",
        "bjet1_e",
        "bjet1_tag_b",
        "bjet1_tag_cvsb",
        "bjet1_tag_cvsl",
        "bjet1_hhbtag",
        "bjet2_px",
        "bjet2_py",
        "bjet2_pz",
        "bjet2_e",
        "bjet2_tag_b",
        "bjet2_tag_cvsb",
        "bjet2_tag_cvsl",
        "bjet2_hhbtag",
        "fatjet_px",
        "fatjet_py",
        "fatjet_pz",
        "fatjet_e",
        "htt_e",
        "htt_px",
        "htt_py",
        "htt_pz",
        "hbb_e",
        "hbb_px",
        "hbb_py",
        "hbb_pz",
        "htthbb_e",
        "htthbb_px",
        "htthbb_py",
        "htthbb_pz",
        "httfatjet_e",
        "httfatjet_px",
        "httfatjet_py",
        "httfatjet_pz",
        "nu1_px",
        "nu1_py",
        "nu1_pz",
        "nu2_px",
        "nu2_py",
        "nu2_pz",
    ]

    continous_features = [
        add_prefix(f, f"{data_prefix}_", ignore_code="_") for f in continous_features
    ]
    categorical_features = [
        add_prefix(f, f"{data_prefix}_", ignore_code="_") for f in categorical_features
    ]

    if debug:
        continous_features = continous_features[:debug_length]
        categorical_features = categorical_features[:debug_length]
    return continous_features, categorical_features


def feature(feature):
    # data_prefix = prefix_map()
    # feature = [add_prefix(f, f"{data_prefix}_", ignore_code="_") for f in feature]
    return tuple(feature)


def feature_vanilla(feature):
    """For Bogdan's original version"""
    data_prefix = prefix_map()
    feature = [add_prefix(f, f"{data_prefix}_", ignore_code="_") for f in feature]
    return tuple(feature)


categorical_features = feature_vanilla(
    [
        "pair_type",
        # "_channel_id",
        "dm1",
        "dm2",
        "vis_tau1_charge",
        "vis_tau2_charge",
        "has_jet_pair",
        "has_fatjet",
    ],
)


continuous_features_vanilla = feature_vanilla(
    [
        "met_px",
        "met_py",
        "met_cov00",
        "met_cov01",
        "met_cov11",
        "vis_tau1_px",
        "vis_tau1_py",
        "vis_tau1_pz",
        "vis_tau1_e",
        "vis_tau2_px",
        "vis_tau2_py",
        "vis_tau2_pz",
        "vis_tau2_e",
        "bjet1_px",
        "bjet1_py",
        "bjet1_pz",
        "bjet1_e",
        "bjet1_tag_b",
        "bjet1_tag_cvsb",
        "bjet1_tag_cvsl",
        "bjet1_hhbtag",
        "bjet2_px",
        "bjet2_py",
        "bjet2_pz",
        "bjet2_e",
        "bjet2_tag_b",
        "bjet2_tag_cvsb",
        "bjet2_tag_cvsl",
        "bjet2_hhbtag",
        "fatjet_px",
        "fatjet_py",
        "fatjet_pz",
        "fatjet_e",
        "htt_e",
        "htt_px",
        "htt_py",
        "htt_pz",
        "hbb_e",
        "hbb_px",
        "hbb_py",
        "hbb_pz",
        "htthbb_e",
        "htthbb_px",
        "htthbb_py",
        "htthbb_pz",
        "httfatjet_e",
        "httfatjet_px",
        "httfatjet_py",
        "httfatjet_pz",
        "nu1_px",
        "nu1_py",
        "nu1_pz",
        "nu2_px",
        "nu2_py",
        "nu2_pz",
    ],
)
continuous_features_detector_observables = feature(
    [
        "pdf_input_vars_reco_higgs_b1_pt",
        "pdf_input_vars_reco_higgs_b1_eta",
        "pdf_input_vars_reco_higgs_b1_phi",
        "pdf_input_vars_reco_higgs_b1_mass",
        "pdf_input_vars_reco_higgs_b2_pt",
        "pdf_input_vars_reco_higgs_b2_eta",
        "pdf_input_vars_reco_higgs_b2_phi",
        "pdf_input_vars_reco_higgs_b2_mass",
        "pdf_input_vars_reco_higgs_tau1_pt",
        "pdf_input_vars_reco_higgs_tau1_eta",
        "pdf_input_vars_reco_higgs_tau1_phi",
        "pdf_input_vars_reco_higgs_tau1_mass",
        "pdf_input_vars_reco_higgs_tau2_pt",
        "pdf_input_vars_reco_higgs_tau2_eta",
        "pdf_input_vars_reco_higgs_tau2_phi",
        "pdf_input_vars_reco_higgs_tau2_mass",
        "pdf_input_vars_reco_top_b1_pt",
        "pdf_input_vars_reco_top_b1_eta",
        "pdf_input_vars_reco_top_b1_phi",
        "pdf_input_vars_reco_top_b1_mass",
        "pdf_input_vars_reco_top_b2_pt",
        "pdf_input_vars_reco_top_b2_eta",
        "pdf_input_vars_reco_top_b2_phi",
        "pdf_input_vars_reco_top_b2_mass",
        "pdf_input_vars_reco_top_tau1_pt",
        "pdf_input_vars_reco_top_tau1_eta",
        "pdf_input_vars_reco_top_tau1_phi",
        "pdf_input_vars_reco_top_tau1_mass",
        "pdf_input_vars_reco_top_tau2_pt",
        "pdf_input_vars_reco_top_tau2_eta",
        "pdf_input_vars_reco_top_tau2_phi",
        "pdf_input_vars_reco_top_tau2_mass",
    ]
)
continuous_features_hard_scattering = feature(
    [
        "pdf_input_vars_reco_higgs_constr_term_b",
        "pdf_input_vars_reco_higgs_constr_term_tau",
        "pdf_input_vars_reco_higgs_cos_theta_cms_h1_b1",
        "pdf_input_vars_reco_higgs_cos_theta_cms_h2_tau_vis1",
        "pdf_input_vars_reco_higgs_cos_theta_h1",
        "pdf_input_vars_reco_higgs_dihiggs_mass",
        "pdf_input_vars_reco_higgs_dihiggs_system_phi",
        "pdf_input_vars_reco_higgs_dihiggs_system_pt",
        "pdf_input_vars_reco_higgs_dihiggs_system_pz",
        "pdf_input_vars_reco_higgs_jac_det",
        "pdf_input_vars_reco_higgs_phi_cms_h1_b1",
        "pdf_input_vars_reco_higgs_phi_cms_h2_tau_vis1",
        "pdf_input_vars_reco_higgs_phi_h1",
        "pdf_input_vars_reco_top_jac_det",
        "pdf_input_vars_reco_top_t1_vis_phi",
        "pdf_input_vars_reco_top_t_vis_y_diff",
        "pdf_input_vars_reco_top_tau1_cos_theta_star_cms_t1_vis",
        "pdf_input_vars_reco_top_tau1_cos_theta_star_cms_wplus",
        "pdf_input_vars_reco_top_tau1_phi",
        "pdf_input_vars_reco_top_tau2_cos_theta_star_cms_t2_vis",
        "pdf_input_vars_reco_top_tau2_cos_theta_star_cms_wminus",
        "pdf_input_vars_reco_top_tau2_phi",
        "pdf_input_vars_reco_top_tt_vis_system_mass",
        "pdf_input_vars_reco_top_tt_vis_system_phi",
        "pdf_input_vars_reco_top_tt_vis_system_pt",
        "pdf_input_vars_reco_top_tt_vis_system_pz",
        # "likelihood_ratio",
    ],
)
continuous_features = continuous_features_detector_observables
# continuous_features = continuous_features_hard_scattering
if int(os.environ["BOGDANS"]):
    continuous_features = continuous_features_vanilla
categorical_features = tuple([])
