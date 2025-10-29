from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={"gen_top.*", "channel_id"},
    produces={"channel_truth.*"},
)
def channel_truth(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Information on how the different decay channels(1-3) defined by channel_id on gen are composed on gen_lvl
    """
    gen_top = events.gen_top
    # pi_zero = tau_minus_children[tau_minus_children.pdgId == 111]
    # pi_plus = tau_minus_children[tau_minus_children.pdgId == 211]
    # pi_minus = tau_minus_children[tau_minus_children.pdgId == -211]
    # K_zero = tau_minus_children[tau_minus_children.pdgId == 311]
    # K_plus = tau_minus_children[tau_minus_children.pdgId == 321]
    # K_minus = tau_minus_children[tau_minus_children.pdgId == -321]

    # Define decay channel maks from gen data
    tau_h_mask = abs(gen_top.w_tau_children.pdgId == 111)
    dilep_mask = ak.all(abs(gen_top.w_children.pdgId[:, :, 0]) >= 11, axis=1)

    semilep_mask = ak.all(ak.concatenate([
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) >= 11, axis=1)[:, None],
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) < 11, axis=1)[:, None],
    ], axis=1), axis=1)
    semilep_e_mask = ak.all(ak.concatenate([
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) == 11, axis=1)[:, None],
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) < 11, axis=1)[:, None],
    ], axis=1), axis=1)
    semilep_mu_mask = ak.all(ak.concatenate([
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) == 13, axis=1)[:, None],
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) < 11, axis=1)[:, None],
    ], axis=1), axis=1)
    semilep_tau_mask = ak.all(ak.concatenate([
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) == 15, axis=1)[:, None],
        ak.any(abs(gen_top.w_children.pdgId[:, :, 0]) < 11, axis=1)[:, None],
    ], axis=1), axis=1)

    full_hadr_mask = ak.all(abs(gen_top.w_children.pdgId[:, :, 0]) < 11, axis=1)

    # Get channel_id columns and check the gen channel distribution
    # channel_id 1: e tau_h
    # channel_id 2: mu tau_h
    # channel_id 3: full hadr
    channel_id = events.channel_id
    semileptonic_channel_id_mask = ak.any(ak.concatenate([
        (channel_id == 1)[:, None], (channel_id == 2)[:, None],
    ], axis=1), axis=1)
    full_hadr_channel_id_mask = channel_id == 3

    # Final columns
    # semilep channels
    sl_ch_id_is_dilep = ak.all(ak.concatenate([
        semileptonic_channel_id_mask[:, None], dilep_mask[:, None],
    ], axis=1), axis=1)
    sl_ch_id_is_dilep = ak.mask(ak.full_like(sl_ch_id_is_dilep, 1, dtype=int), sl_ch_id_is_dilep)
    # sl_ch_id_is_dilep = ak.where(
    #     sl_ch_id_is_dilep,
    #     ak.full_like(sl_ch_id_is_dilep, 1, dtype=int),
    #     ak.full_like(sl_ch_id_is_dilep, 0, dtype=int),
    # )
    sl_ch_id_is_semilep = ak.all(ak.concatenate([
        semileptonic_channel_id_mask[:, None], semilep_mask[:, None],
    ], axis=1), axis=1)
    sl_ch_id_is_semilep = ak.mask(ak.full_like(sl_ch_id_is_semilep, 2, dtype=int), sl_ch_id_is_semilep)
    # sl_ch_id_is_semilep = ak.where(
    #     sl_ch_id_is_semilep,
    #     ak.full_like(sl_ch_id_is_semilep, 1, dtype=int),
    #     ak.full_like(sl_ch_id_is_semilep, 0, dtype=int),
    # )
    sl_ch_id_is_full_hadr = ak.all(ak.concatenate([
        semileptonic_channel_id_mask[:, None], full_hadr_mask[:, None],
    ], axis=1), axis=1)
    sl_ch_id_is_full_hadr = ak.mask(ak.full_like(sl_ch_id_is_full_hadr, 3, dtype=int), sl_ch_id_is_full_hadr)
    # sl_ch_id_is_full_hadr = ak.where(
    #     sl_ch_id_is_full_hadr,
    #     ak.full_like(sl_ch_id_is_full_hadr, 1, dtype=int),
    #     ak.full_like(sl_ch_id_is_full_hadr, 0, dtype=int),
    # )
    # Full hadr channel
    fh_ch_id_is_dilep = ak.all(ak.concatenate([
        full_hadr_channel_id_mask[:, None], dilep_mask[:, None],
    ], axis=1), axis=1)
    fh_ch_id_is_dilep = ak.mask(ak.full_like(fh_ch_id_is_dilep, 1, dtype=int), fh_ch_id_is_dilep)
    # fh_ch_id_is_dilep = ak.where(
    #     fh_ch_id_is_dilep,
    #     ak.full_like(fh_ch_id_is_dilep, 1, dtype=int),
    #     ak.full_like(fh_ch_id_is_dilep, 0, dtype=int),
    # )
    fh_ch_id_is_semilep = ak.all(ak.concatenate([
        full_hadr_channel_id_mask[:, None], semilep_mask[:, None],
    ], axis=1), axis=1)
    fh_ch_id_is_semilep = ak.mask(ak.full_like(fh_ch_id_is_semilep, 2, dtype=int), fh_ch_id_is_semilep)
    # fh_ch_id_is_semilep = ak.where(
    #     fh_ch_id_is_semilep,
    #     ak.full_like(fh_ch_id_is_semilep, 1, dtype=int),
    #     ak.full_like(fh_ch_id_is_semilep, 0, dtype=int),
    # )
    fh_ch_id_is_full_hadr = ak.all(ak.concatenate([
        full_hadr_channel_id_mask[:, None], full_hadr_mask[:, None],
    ], axis=1), axis=1)
    fh_ch_id_is_full_hadr = ak.mask(ak.full_like(fh_ch_id_is_full_hadr, 3, dtype=int), fh_ch_id_is_full_hadr)
    # fh_ch_id_is_full_hadr = ak.where(
    #     fh_ch_id_is_full_hadr,
    #     ak.full_like(fh_ch_id_is_full_hadr, 1, dtype=int),
    #     ak.full_like(fh_ch_id_is_full_hadr, 0, dtype=int),
    # )
    sl_ch_id_truth = ak.concatenate([sl_ch_id_is_dilep[:, None],
                                     sl_ch_id_is_semilep[:, None],
                                     sl_ch_id_is_full_hadr[:, None],
                                     ], axis=1)
    sl_ch_id_truth = ak.max(sl_ch_id_truth, axis=1)
    fh_ch_id_truth = ak.concatenate([fh_ch_id_is_dilep[:, None],
                                     fh_ch_id_is_semilep[:, None],
                                     fh_ch_id_is_full_hadr[:, None],
                                     ], axis=1)
    fh_ch_id_truth = ak.max(fh_ch_id_truth, axis=1)
    # sl_ch_id_truth = ak.fill_none(sl_ch_id_truth, 0, axis=1)
    # fh_ch_id_truth = ak.fill_none(fh_ch_id_truth, 0, axis=1)
    channel_truth = ak.zip({
        # "sl_ch_id_is_dilep": sl_ch_id_is_dilep,
        # "sl_ch_id_is_semilep": sl_ch_id_is_semilep,
        # "sl_ch_id_is_full_hadr": sl_ch_id_is_full_hadr,
        # "fh_ch_id_is_dilep": fh_ch_id_is_dilep,
        # "fh_ch_id_is_semilep": fh_ch_id_is_semilep,
        # "fh_ch_id_is_full_hadr": fh_ch_id_is_full_hadr,
        "sl_ch_id_truth": sl_ch_id_truth,
        "fh_ch_id_truth": fh_ch_id_truth,
    }, with_name="channel_truth")

    events = set_ak_column(events, "channel_truth", channel_truth)
    return events
