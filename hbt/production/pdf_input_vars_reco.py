from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn
import numpy as np
# import vector

ak = maybe_import("awkward")


@producer(
    uses={"higgs_family.*", "HHBJet", attach_coffea_behavior},
    produces={"pdf_input_vars_reco_higgs.*"},
)
def create_pdf_input_vars_reco_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_reco_higgs" that contains the inputs for the signal likelihood. Only fully
    matched events are stored.
    """

    # Get relevant columns and attach coffea_behavior
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
        "tau_leptonic_decay_products": {
            "type_name": "GenParticle", "check_attr": "metric_table", "skip_fields": "*Idx*G",
        },
    })
    events = attach_coffea_behavior_fn(events, collections={"HHBJet": {"type_name": "LorentzVector"},
        "Tau": {"type_name": "LorentzVector"}})

    # Create H_bb
    b_jets = events.HHBJet
    H_bb = b_jets.sum(axis=1)

    # Match reco bjets to gen b's: Only fully matched events atm
    gen_bs = higgs_family.bottoms
    b_jet_0_mask = gen_bs.deltaR(b_jets[:, 0]) < 0.4
    b_jet_1_mask = gen_bs.deltaR(b_jets[:, 1]) < 0.4
    b_jet_0 = b_jets[b_jet_0_mask]
    b_jet_1 = b_jets[b_jet_1_mask]
    matched_b_jets = ak.concatenate([b_jet_0, b_jet_1], axis=1)
    fully_matched_events_mask = ak.num(matched_b_jets, axis=1) == 2
    matched_b_jets = ak.mask(matched_b_jets, fully_matched_events_mask)
    matched_b_jets = ak.pad_none(matched_b_jets, 2, axis=1, clip=True)

    # Match tau's: Only full hadronic events atm, and even here the tau neutrinos are missing
    gen_taus = higgs_family.taus
    tau_jets = events.Tau
    full_hadr_decay_mask = ak.num(events.Tau, axis=1) == 2
    relevant_taus = ak.mask(tau_jets, full_hadr_decay_mask)
    tau_0_mask = relevant_taus.deltaR(gen_taus[:, 0]) < 0.4
    tau_1_mask = relevant_taus.deltaR(gen_taus[:, 1]) < 0.4
    tau_0 = relevant_taus[tau_0_mask]
    tau_1 = relevant_taus[tau_1_mask]
    matched_taus = ak.concatenate([tau_0, tau_1], axis=1)
    matched_taus = ak.pad_none(matched_taus, 2, axis=1, clip=True)

    # Add tau neutrinos from gen collection for now
    # tau_nu_0 = ak.mask(higgs_family.tau_leptonic_decay_products[:, 0], ak.any(tau_0_mask, axis=1))[:, 0]
    # tau_nu_1 = higgs_family.tau_leptonic_decay_products[:, 0][tau_1_mask]
    # Build H_tautau
    H_tautau = matched_taus.sum(axis=1)

    # Coompute the pdf inputs
    dihiggs_system = H_bb.add(H_tautau)
    dihiggs_mass = dihiggs_system.mass
    dihiggs_system_pt = dihiggs_system.pt
    dihiggs_system_pz = dihiggs_system.pz
    dihiggs_system_phi = dihiggs_system.phi

    h1 = H_bb
    h1_cms_dihiggs = h1.boostCM_of(dihiggs_system)
    theta_h1 = dihiggs_system.deltaangle(h1_cms_dihiggs)       # angle between h1 in dihiggs cms and dihiggs in
    # lab system. always positive, do we want that?
    cos_theta_h1 = np.cos(theta_h1)
    phi_h1 = h1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h2 = H_tautau
    tau1 = higgs_family.taus[:, 0]
    tau1_cms_h2 = tau1.boostCM_of(h2.boostvec)
    theta_cms_h2_tau1 = h2.deltaangle(tau1_cms_h2)   # angle between tau1 in cms of h2 and h2 in lab system
    cos_theta_cms_h2_tau1 = np.cos(theta_cms_h2_tau1)
    phi_cms_h2_tau1 = tau1_cms_h2.phi   # phi of tau1 in h2's cms
    # theta_star_tau_tau = higgs_family[:,2,1].boostCM_of(h2.boostvec).deltaangle(tau1_cms_h2)   # angle between
    # tau1 and tau2 in h2's cms

    # boost b1 into cms of hbb
    b1 = higgs_family.bottoms[:, 0]
    b1_cms_h1 = b1.boostCM_of(h1.boostvec)
    theta_cms_h1_b1 = h1.deltaangle(b1_cms_h1)     # angle between b1 in cms of h1 and h1 in lab system
    cos_theta_cms_h1_b1 = np.cos(theta_cms_h1_b1)
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

    events = set_ak_column(events, "pdf_input_vars_reco_higgs", pdf_input_vars)

    return events
