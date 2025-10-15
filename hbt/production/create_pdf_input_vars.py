# coding: utf-8

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn
from columnflow.columnar_util import EMPTY_FLOAT
import numpy as np
# import vector

ak = maybe_import("awkward")


def signed_cos_deltaangle(a, b):
    """
    Returns the signed angle between two vectors a and b.
    """
    ax, ay, az = a.px, a.py, a.pz
    bx, by, bz = b.px, b.py, b.pz
    a_times_b = ax * bx + ay * by + az * bz
    abs_a = np.sqrt(ax**2 + ay**2 + az**2)
    abs_b = np.sqrt(bx**2 + by**2 + bz**2)
    return a_times_b / (abs_a * abs_b)


def weird_conversion(array: ak.Array, axis: int = 0) -> ak.Array:
    array = ak.from_numpy(ak.to_numpy(array).astype(np.float64))
    array = ak.fill_none(array, EMPTY_FLOAT, axis=axis)
    return array


@producer(
    uses={"higgs_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars.*"},
)
def create_pdf_input_vars_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars" that stores the PDF input variables for the Higgs decay products.
    """
    # Get higgs_family column and attach coffea behavior
    # events = self[attach_coffea_behavior](events, collections={"higgs_family": {"type_name": "GenParticle",
    #     "check_attr": "metric_table", "skip_fields": "*Idx*G"}}, **kwargs)

    higgs_family = attach_coffea_behavior_fn(events.higgs_family, collections={
        "bottoms": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "taus": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "higgs": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
    })
    # Define input vars
    dihiggs_system = higgs_family.higgs.sum(axis=1)
    dihiggs_mass = dihiggs_system.mass
    dihiggs_system_pt = dihiggs_system.pt
    dihiggs_system_pz = dihiggs_system.pz
    dihiggs_system_phi = dihiggs_system.phi
    # boost h1 into cms of dihiggs system
    h1 = higgs_family.higgs[:, 0]
    h1_cms_dihiggs = h1.boostCM_of(dihiggs_system)
    # angle between h1 in dihiggs cms and dihiggs in lab system
    cos_theta_h1 = signed_cos_deltaangle(h1_cms_dihiggs, dihiggs_system)
    phi_h1 = h1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h2 = higgs_family.higgs[:, 1].boostCM_of(dihiggs_system)
    tau1 = higgs_family.taus[:, 0]
    tau1_cms_h2 = tau1.boostCM_of(h2.boostvec)
    # theta_cms_h2_tau1 = h2.deltaangle(tau1_cms_h2)   # angle between tau1 in cms of h2 and h2 in lab system
    cos_theta_cms_h2_tau1 = signed_cos_deltaangle(tau1_cms_h2, h2)
    phi_cms_h2_tau1 = tau1_cms_h2.phi   # phi of tau1 in h2's cms

    # boost b1 into cms of hbb
    b1 = higgs_family.bottoms[:, 0]
    b1_cms_h1 = b1.boostCM_of(h1_cms_dihiggs.boostvec)
    # angle between b1 in cms of h1 and h1 in lab system
    cos_theta_cms_h1_b1 = signed_cos_deltaangle(b1_cms_h1, h1_cms_dihiggs)
    phi_cms_h1_b1 = b1_cms_h1.phi   # phi of b1 in h1's cms

    pdf_input_vars = ak.zip({"dihiggs_mass": dihiggs_mass,
                             "dihiggs_system_pt": dihiggs_system_pt,
                             "dihiggs_system_pz": dihiggs_system_pz,
                             "dihiggs_system_phi": dihiggs_system_phi,
                             "cos_theta_h1": cos_theta_h1,
                             "phi_h1": phi_h1,
                             "cos_theta_cms_h2_tau1": cos_theta_cms_h2_tau1,
                             "phi_cms_h2_tau1": phi_cms_h2_tau1,
                             "cos_theta_cms_h1_b1": cos_theta_cms_h1_b1,
                             "phi_cms_h1_b1": phi_cms_h1_b1}, with_name="pdf_input_vars")

    events = set_ak_column(events, "pdf_input_vars", pdf_input_vars)

    return events


@producer(
    uses={"top_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars.*"},
)
def create_pdf_input_vars_top(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars" that stores the PDF input variables for the Higgs decay products.
    """
    # Get top_family column and attach coffea behavior
    events = self[attach_coffea_behavior](events, collections={"top_family": {"type_name": "GenParticle",
        "check_attr": "metric_table", "skip_fields": "*Idx*G"}}, **kwargs)
    top_family = events.top_family

    input_vars_dict = {}
    # Define inputs for each factorization step
    s_hat_system = top_family[:, 0].sum(axis=1)
    input_vars_dict["s_hat"] = s_hat_system.mass
    input_vars_dict["s_hat_system_pt"] = s_hat_system.pt
    input_vars_dict["s_hat_system_pz"] = s_hat_system.pz
    input_vars_dict["s_hat_system_phi"] = s_hat_system.phi

    top = top_family[:, 0, 0].boostCM_of(s_hat_system.boostvec)
    input_vars_dict["phi_t"] = top.phi
    input_vars_dict["cos_theta_t"] = signed_cos_deltaangle(top, s_hat_system)
    for i in range(2):
        input_vars_dict["W_t" + str(i)] = top_family[:, 2, i].boostCM_of(
            top_family[:, 0, i].boostCM_of(s_hat_system).boostvec,
        )
        input_vars_dict["W_t" + str(i) + "_phi"] = input_vars_dict["W_t" + str(i)].phi
        input_vars_dict["W_t" + str(i) + "cos_theta"] = signed_cos_deltaangle(
            input_vars_dict["W_t" + str(i)],
            top_family[:, 0, i].boostCM_of(s_hat_system),
        )

    # want child of W that is no neutrino: Either q qbar pair or lepton (11,13,15)
    # Which q to take?
    q_or_antilep_t0 = top_family[:, [3, 6], 0]
    q_or_antilep_t0_mask = q_or_antilep_t0.pdgId != EMPTY_FLOAT
    W_t0_child = q_or_antilep_t0[q_or_antilep_t0_mask].boostCM_of(top_family[:, 2, 0].boostCM_of(s_hat_system).boostvec)
    q_or_antilep_t1 = top_family[:, [3, 5], 1]
    q_or_antilep_t1_mask = q_or_antilep_t1.pdgId != EMPTY_FLOAT
    W_t1_child = q_or_antilep_t1[q_or_antilep_t1_mask].boostCM_of(top_family[:, 2, 1].boostCM_of(s_hat_system).boostvec)
    for i in range(2):
        input_vars_dict["W_t" + str(i) + "_child_phi"] = eval(f"W_t{i}_child").phi
        input_vars_dict["W_t" + str(i) + "_child_cos_theta"] = eval(
            f"signed_cos_deltaangle(W_t{i}_child, input_vars_dict['W_t' + str(i)])",
        )

    pdf_input_vars = ak.zip(input_vars_dict, with_name="pdf_input_vars")
    events = set_ak_column(events, "pdf_input_vars", pdf_input_vars)
    W_t0_child = W_t0_child
    W_t1_child = W_t1_child
    return events


@producer(
    uses={"top_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars_top_ditau.*"},
)
def create_pdf_input_vars_top_ditau(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_top_ditau" that stores the PDF inputs for the top pdf in the ditau case
    """

    # Get top_family column and attach coffea behaviour
    top_family = attach_coffea_behavior_fn(events.top_family, collections={
        "bottoms": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "leps": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "antileps": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "neutrinos": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "antineutrinos": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
    })

    # Extract relevant particles
    tau_mask = top_family.leps.pdgId == 15
    antitau_mask = top_family.antileps.pdgId == -15
    ditau_mask = ak.all(ak.concatenate([
        ak.any(tau_mask, axis=1)[:, None], ak.any(antitau_mask, axis=1)[:, None],
    ], axis=1), axis=1)
    # ditau_decays = ak.mask(top_family, ditau_mask)
    # bs = ak.mask(ditau_decays.bottoms, top_family.bottoms.pdgId == 5)
    # bs = ak.where(ditau_mask, top_family.bottoms[:, 0], ak.full_like(top_family.bottoms[:, 0], EMPTY_FLOAT))
    # bs = ditau_decays.bottoms[:, 0][:, None]
    bs = ak.mask(top_family, ditau_mask).bottoms[:, 0][:, None]
    antibs = ak.mask(top_family, ditau_mask).bottoms[:, 1][:, None]
    tau_vis = ak.mask(top_family, ditau_mask).leps - ak.mask(top_family, ditau_mask).neutrinos
    antitau_vis = ak.mask(top_family, ditau_mask).antileps - ak.mask(top_family, ditau_mask).antineutrinos

    tau_vis_antib = tau_vis.add(antibs)
    antitau_vis_b = antitau_vis.add(bs)

    # Build inputs
    M_2tau_vis_2b = tau_vis_antib.add(antitau_vis_b).absolute()

    y_tau_vis_antib = 1 / 2 * np.log(
        (tau_vis_antib.energy + tau_vis_antib.pz) / (tau_vis_antib.energy - tau_vis_antib.pz))
    y_tau_vis_antib = np.nan_to_num(y_tau_vis_antib)    # , nan=EMPTY_FLOAT)
    y_tau_vis_antib = ak.firsts(y_tau_vis_antib)
    y_antitau_vis_b = 1 / 2 * np.log((antitau_vis_b.energy + antitau_vis_b.pz) /
        (antitau_vis_b.energy - antitau_vis_b.pz))
    y_antitau_vis_b = ak.firsts(np.nan_to_num(y_antitau_vis_b))

    pt_tau_vis_antib = tau_vis_antib.pt
    pt_antitau_vis_b = antitau_vis_b.pt

    M_tau_vis_antib = tau_vis_antib.absolute()
    M_antitau_vis_b = antitau_vis_b.absolute()

    # Same inputs, but with tau replacing tau_vis to estimate impact of missing neutrinos
    taus = ak.mask(top_family, ditau_mask).leps
    antitaus = ak.mask(top_family, ditau_mask).antileps

    tau_antib = taus.add(antibs)
    antitau_b = antitaus.add(bs)

    M_2tau_2b = tau_antib.add(antitau_b).absolute()

    y_tau_antib = ak.firsts(
        np.nan_to_num(1 / 2 * np.log((tau_antib.energy + tau_antib.pz) / (tau_antib.energy - tau_antib.pz))))
    y_antitau_b = ak.firsts(
        np.nan_to_num(1 / 2 * np.log((antitau_b.energy + antitau_b.pz) / (antitau_b.energy - antitau_b.pz))))

    pt_tau_antib = tau_antib.pt
    pt_antitau_b = antitau_b.pt

    M_tau_antib = tau_antib.absolute()
    M_antitau_b = antitau_b.absolute()

    # Ratio of these inputs
    four_part_mass_ratio = M_2tau_vis_2b * 1 / M_2tau_2b
    # rapidity_ratio_tau = ak.to_numpy(y_tau_vis_b) * 1 / ak.to_numpy(y_tau_b)
    pt_ratio_tau = pt_tau_vis_antib / pt_tau_antib
    two_part_mass_ratio_tau = M_tau_vis_antib / M_tau_antib

    # Remove Nones
    M_2tau_vis_2b = weird_conversion(M_2tau_vis_2b, axis=1)
    y_tau_vis_antib = weird_conversion(y_tau_vis_antib)
    y_antitau_vis_b = weird_conversion(y_antitau_vis_b)
    pt_tau_vis_antib = weird_conversion(pt_tau_vis_antib, axis=1)
    pt_antitau_vis_b = weird_conversion(pt_antitau_vis_b, axis=1)
    M_tau_vis_antib = weird_conversion(M_tau_vis_antib, axis=1)
    M_antitau_vis_b = weird_conversion(M_antitau_vis_b, axis=1)
    M_2tau_2b = weird_conversion(M_2tau_2b, axis=1)
    y_tau_antib = weird_conversion(y_tau_antib)
    y_antitau_b = weird_conversion(y_antitau_b)
    pt_tau_antib = weird_conversion(pt_tau_antib, axis=1)
    pt_antitau_b = weird_conversion(pt_antitau_b, axis=1)
    M_tau_antib = weird_conversion(M_tau_antib, axis=1)
    M_antitau_b = weird_conversion(M_antitau_b, axis=1)
    four_part_mass_ratio = weird_conversion(four_part_mass_ratio, axis=1)
    # rapidity_ratio_tau = weird_conversion(rapidity_ratio_tau)
    pt_ratio_tau = weird_conversion(pt_ratio_tau, axis=1)
    two_part_mass_ratio_tau = weird_conversion(two_part_mass_ratio_tau, axis=1)

    pdf_input_vars_top_ditau = ak.zip({
        "M_2tau_vis_2b": M_2tau_vis_2b,
        "y_tau_vis_antib": y_tau_vis_antib,
        "y_antitau_vis_b": y_antitau_vis_b,
        "pt_tau_vis_antib": pt_tau_vis_antib,
        "pt_antitau_vis_b": pt_antitau_vis_b,
        "M_tau_vis_antib": M_tau_vis_antib,
        "M_antitau_vis_b": M_antitau_vis_b,
        "M_2tau_2b": M_2tau_2b,
        "y_tau_antib": y_tau_antib,
        "y_antitau_b": y_antitau_b,
        "pt_tau_antib": pt_tau_antib,
        "pt_antitau_b": pt_antitau_b,
        "M_tau_antib": M_tau_antib,
        "M_antitau_b": M_antitau_b,
        "four_part_mass_ratio": four_part_mass_ratio,
        # "rapidity_ratio_tau": rapidity_ratio_tau,
        "pt_ratio_tau": pt_ratio_tau,
        "two_part_mass_ratio_tau": two_part_mass_ratio_tau,
    }, with_name="pdf_input_vars_top_ditau")
    events = set_ak_column(events, "pdf_input_vars_top_ditau", pdf_input_vars_top_ditau)
    return events


@producer(
    uses={"higgs_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars_top_ditau_higgs.*"},
)
def create_pdf_input_vars_top_ditau_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_top_ditau" that stores the PDF inputs for the top pdf in the ditau case
    """
    # attach coffea behaviour
    higgs_family = attach_coffea_behavior_fn(events.higgs_family, collections={
        "bottoms": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "taus": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
        "tau_leptonic_decay_products": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
    })
    # from IPython import embed
    # embed(header="higgs ditau stuff")
    bs = higgs_family.bottoms[:, 1][:, None]
    antibs = higgs_family.bottoms[:, 0][:, None]
    tau_vis = higgs_family.taus[:, 1] - higgs_family.tau_leptonic_decay_products[:, 0, 1]
    antitau_vis = higgs_family.taus[:, 0] - higgs_family.tau_leptonic_decay_products[:, 0, 0]
    taus = higgs_family.taus[:, 1]
    antitaus = higgs_family.taus[:, 0]

    tau_vis_antib = tau_vis[:, None].add(antibs)
    antitau_vis_b = antitau_vis[:, None].add(bs)

    # Build inputs
    M_2tau_vis_2b = tau_vis_antib.add(antitau_vis_b).absolute()

    y_tau_vis_antib = 1 / 2 * np.log(
        (tau_vis_antib.energy + tau_vis_antib.pz) / (tau_vis_antib.energy - tau_vis_antib.pz))
    y_tau_vis_antib = np.nan_to_num(y_tau_vis_antib)    # , nan=EMPTY_FLOAT)
    y_tau_vis_antib = ak.firsts(y_tau_vis_antib)
    y_antitau_vis_b = 1 / 2 * np.log((antitau_vis_b.energy + antitau_vis_b.pz) /
        (antitau_vis_b.energy - antitau_vis_b.pz))
    y_antitau_vis_b = ak.firsts(np.nan_to_num(y_antitau_vis_b))

    pt_tau_vis_antib = tau_vis_antib.pt
    pt_antitau_vis_b = antitau_vis_b.pt

    M_tau_vis_antib = tau_vis_antib.absolute()
    M_antitau_vis_b = antitau_vis_b.absolute()

    # Same inputs, but with tau replacing tau_vis to estimate impact of missing neutrinos
    tau_antib = taus[:, None].add(antibs)
    antitau_b = antitaus[:, None].add(bs)

    M_2tau_2b = tau_antib.add(antitau_b).absolute()

    y_tau_antib = ak.firsts(
        np.nan_to_num(1 / 2 * np.log((tau_antib.energy + tau_antib.pz) / (tau_antib.energy - tau_antib.pz))))
    y_antitau_b = ak.firsts(
        np.nan_to_num(1 / 2 * np.log((antitau_b.energy + antitau_b.pz) / (antitau_b.energy - antitau_b.pz))))

    pt_tau_antib = tau_antib.pt
    pt_antitau_b = antitau_b.pt

    M_tau_antib = tau_antib.absolute()
    M_antitau_b = antitau_b.absolute()

    # Ratio of these inputs
    four_part_mass_ratio = M_2tau_vis_2b * 1 / M_2tau_2b
    # rapidity_ratio_tau = ak.to_numpy(y_tau_vis_b) * 1 / ak.to_numpy(y_tau_b)
    pt_ratio_tau = pt_tau_vis_antib / pt_tau_antib
    two_part_mass_ratio_tau = M_tau_vis_antib / M_tau_antib

    pdf_input_vars_top_ditau_higgs = ak.zip({
        "M_2tau_vis_2b": M_2tau_vis_2b,
        "y_tau_vis_antib": y_tau_vis_antib,
        "y_antitau_vis_b": y_antitau_vis_b,
        "pt_tau_vis_antib": pt_tau_vis_antib,
        "pt_antitau_vis_b": pt_antitau_vis_b,
        "M_tau_vis_antib": M_tau_vis_antib,
        "M_antitau_vis_b": M_antitau_vis_b,
        "M_2tau_2b": M_2tau_2b,
        "y_tau_antib": y_tau_antib,
        "y_antitau_b": y_antitau_b,
        "pt_tau_antib": pt_tau_antib,
        "pt_antitau_b": pt_antitau_b,
        "M_tau_antib": M_tau_antib,
        "M_antitau_b": M_antitau_b,
        "four_part_mass_ratio": four_part_mass_ratio,
        # "rapidity_ratio_tau": rapidity_ratio_tau,
        "pt_ratio_tau": pt_ratio_tau,
        "two_part_mass_ratio_tau": two_part_mass_ratio_tau,
    }, with_name="pdf_input_vars_top_ditau_higgs")
    events = set_ak_column(events, "pdf_input_vars_top_ditau_higgs", pdf_input_vars_top_ditau_higgs)

    return events
