from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column

np = maybe_import("numpy")
ak = maybe_import("awkward")


@producer(
    uses={"gen_top.*", "channel_id"},
    produces={"channel_truth.*", "channel_truth_sums.*"},
)
def channel_truth(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """Information on how the different decay channels(1-3) defined by channel_id on gen are composed on gen_lvl
    """
    # from IPython import embed
    # embed(header="channel_truth")
    gen_top = events.gen_top
    # pi_zero = tau_minus_children[tau_minus_children.pdgId == 111]
    # pi_plus = tau_minus_children[tau_minus_children.pdgId == 211]
    # pi_minus = tau_minus_children[tau_minus_children.pdgId == -211]
    # K_zero = tau_minus_children[tau_minus_children.pdgId == 311]
    # K_plus = tau_minus_children[tau_minus_children.pdgId == 321]
    # K_minus = tau_minus_children[tau_minus_children.pdgId == -321]

    # Define decay channel maks from gen data
    print(set(ak.sort(ak.ravel(gen_top.w_tau_children.pdgId))))
    # tau_mask = ak.any(ak.any(abs(gen_top.w_children.pdgId) == 15, axis=2), axis=1)
    tau_h_mask = ak.any(ak.any(ak.any(abs(gen_top.w_tau_children.pdgId) >= 111, axis=3), axis=2), axis=1)
    tau_h_mask = tau_h_mask
    tau_l_mask = ak.any(ak.any(ak.any(abs(gen_top.w_tau_children.pdgId) < 16, axis=3), axis=2), axis=1)
    tau_l_mask = tau_l_mask
    e_mask = ak.any(ak.any(abs(gen_top.w_children.pdgId) == 11, axis=2), axis=1)
    e_mask = e_mask
    mu_mask = ak.any(ak.any(abs(gen_top.w_children.pdgId) == 13, axis=2), axis=1)
    mu_mask = mu_mask
    qq_mask = ak.any(ak.any(abs(gen_top.w_children.pdgId) <= 6, axis=2), axis=1)
    qq_mask = qq_mask

    mask_dict = {}
    idx = 0
    part_list = ["e", "mu", "qq", "tau_l", "tau_h"]
    part_list2 = part_list

    for part1 in part_list:
        if idx == 0:
            for part2 in part_list:
                if part1 == part2:
                    continue
                mask_dict[f"{part1}_{part2}_mask"] = ak.all(ak.concatenate([eval(part1 + "_mask")[:, None],
                eval(part2 + "_mask")[:, None]], axis=1), axis=1)
            idx += 1
        else:
            part_list2 = part_list[idx:]
            for part2 in part_list2:
                if part1 == part2:
                    continue
                mask_dict[f"{part1}_{part2}_mask"] = ak.all(ak.concatenate([eval(part1 + "_mask")[:, None],
                eval(part2 + "_mask")[:, None]], axis=1), axis=1)
            idx += 1

    # mask_dict["gen_ch1_mask"] = gen_ch1_mask
    # mask_dict["gen_ch2_mask"] = gen_ch2_mask
    # mask_dict["gen_ch3_mask"] = gen_ch3_mask
    mask_dict["e_e_mask"] = ak.all(ak.concatenate([
        ak.any(ak.any(gen_top.w_children.pdgId == 11, axis=2), axis=1)[:, None],
        ak.any(ak.any(gen_top.w_children.pdgId == -11, axis=2), axis=1)[:, None],
    ], axis=1), axis=1)
    mask_dict["mu_mu_mask"] = ak.all(ak.concatenate([
        ak.any(ak.any(gen_top.w_children.pdgId == 13, axis=2), axis=1)[:, None],
        ak.any(ak.any(gen_top.w_children.pdgId == -13, axis=2), axis=1)[:, None],
    ], axis=1), axis=1)
    mask_dict["tau_l_tau_l_mask"] = ak.all(
        ak.any(ak.any(abs(gen_top.w_tau_children.pdgId) < 16, axis=3), axis=2), axis=1,
    )
    mask_dict["qq_qq_mask"] = ak.all(ak.any(abs(gen_top.w_children.pdgId) <= 6, axis=2), axis=1)
    mask_dict["tau_h_tau_h_mask"] = ak.all(
        ak.any(ak.any(abs(gen_top.w_tau_children.pdgId) >= 111, axis=3), axis=2), axis=1)
    # Get channel_id columns and check the gen channel distribution
    # channel_id 1: e tau_h
    # channel_id 2: mu tau_h
    # channel_id 3: full hadr
    ch_id = events.channel_id
    ch_id1_mask = ch_id == 1
    ch_id2_mask = ch_id == 2
    ch_id3_mask = ch_id == 3
    ch_id456_mask = ch_id > 3
    ch_id1_mask, ch_id2_mask, ch_id3_mask = ch_id1_mask, ch_id2_mask, ch_id3_mask
    # if not ak.any(ak.any(ak.concatenate([
    #         ch_id1_mask[:, None], ch_id2_mask[:, None], ch_id3_mask[:, None], ch_id456_mask[:, None]],
    #         axis=1), axis=1), axis=0):
    #     exit(1)
    #     from IPython import embed
    #     embed("no correct channel id definition")

    channel_truth_dict = {}
    for id in ["ch_id1", "ch_id2", "ch_id3"]:
        for key in mask_dict.keys():
            channel_truth_dict[f"{id}_{key}"] = ak.all(ak.concatenate([eval(f"{id}_mask")[:, None],
            mask_dict[key][:, None]], axis=1), axis=1)
        channel_truth_dict[f"{id}_mask"] = eval(f"{id}_mask")
    channel_truth_dict["ch_id456_mask"] = ch_id456_mask
    channel_truth_sums = {}
    # idx = 0
    # for key in channel_truth_dict.keys():
    #     if idx == 0:
    #         any_true = ak.Array(channel_truth_dict[key])
    #         idx += 1
    #     if idx == 1:
    #         any_true = ak.concatenate([any_true[:, None], channel_truth_dict[key][:, None]], axis=1)
    #         idx += 1
    #     else:
    #         any_true = ak.concatenate([any_true, channel_truth_dict[key][:, None]], axis=1)
    # any_true = ak.any(any_true, axis=1)
    # if not ak.all(any_true, axis=0):
    #     exit(2)
    #     from IPython import embed
    #     embed("nothing is true")
    # sum = 0
    for key in channel_truth_dict.keys():
        channel_truth_sums[key] = ak.sum(channel_truth_dict[key])
    channel_truth_sums["ch_id456_mask"] = ak.sum(ch_id456_mask)
    # Debugging / Validation
    #     if (key != "ch_id1_mask" and key != "ch_id2_mask" and key != "ch_id3_mask"):
    #         sum += ak.sum(channel_truth_dict[key])
    # print(sum)
    # none_true = np.array([])
    # for idx in range(len(events)):
    #     if idx % 1000 == 0:
    #         print(idx)
    #     true_list = []
    #     for key in channel_truth_dict.keys():
    #         if (key != "ch_id1_mask" and key != "ch_id2_mask" and key != "ch_id3_mask"):
    #             true_list.append(channel_truth_dict[key][idx])
    #     if not ak.any(true_list, axis=0):
    #         none_true = np.append(none_true, idx)
    # wrong_ch_id_list = []
    # for idx in none_true:
    #     wrong_ch_id_list.append(events[int(idx)].channel_id)
    channel_truth = ak.zip(channel_truth_dict, with_name="channel_truth")
    channel_truth_sums = ak.zip(channel_truth_sums, with_name="channel_truth_sums")
    events = set_ak_column(events, "channel_truth", channel_truth)
    events = set_ak_column(events, "channel_truth_sums", channel_truth_sums)
    return events
