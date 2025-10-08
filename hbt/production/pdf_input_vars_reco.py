from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn, EMPTY_FLOAT
from hbt.production.create_pdf_input_vars import signed_cos_deltaangle
import numpy as np
import vector

ak = maybe_import("awkward")


@producer(
    uses={"higgs_family.*", "HHBJet", "reg_dnn_nu*", attach_coffea_behavior},
    produces={"pdf_input_vars_reco_higgs.*"},
)
def create_pdf_input_vars_reco_higgs(
    self: Producer, events: ak.Array, **kwargs,
) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_reco_higgs" that contains the inputs for the signal likelihood. Only fully
    matched events are stored.
    """
    # select the correct channels: 1,2,3
    ch_id_mask = events.channel_id < 4
    higgs_family = ak.mask(events, ch_id_mask)

    # Get relevant columns and attach coffea_behavior
    higgs_family = attach_coffea_behavior_fn(
        events.higgs_family,
        collections={
            "bottoms": {
                "type_name": "GenParticle",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            },
            "taus": {
                "type_name": "GenParticle",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            },
            "higgs": {
                "type_name": "GenParticle",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            },
            "tau_leptonic_decay_products": {
                "type_name": "GenParticle",
                "check_attr": "metric_table",
                "skip_fields": "*Idx*G",
            },
        },
    )
    events = attach_coffea_behavior_fn(
        events,
        collections={
            "HHBJet": {"type_name": "LorentzVector"},
            "Tau": {"type_name": "LorentzVector"},
        },
    )

    # Get tau neutrino momenta and construct vectors
    nu_dict = {}
    for idx in range(2):
        for direction in ("px", "py", "pz"):
            nu_dict[f"nu_{idx}_{direction}"] = eval(
                f"events.reg_dnn_nu{idx + 1}_{direction}",
            )
        nu_dict[f"energy_nu_{idx}"] = np.sqrt(
            nu_dict[f"nu_{idx}_px"] ** 2 + nu_dict[f"nu_{idx}_py"] ** 2 + nu_dict[f"nu_{idx}_pz"] ** 2,
        )
        nu_dict[f"nu_{idx}"] = vector.zip(
            {
                "px": nu_dict[f"nu_{idx}_px"],
                "py": nu_dict[f"nu_{idx}_py"],
                "pz": nu_dict[f"nu_{idx}_pz"],
                "energy": nu_dict[f"energy_nu_{idx}"],
            },
        )
    tau_neutrinos = ak.concatenate(
        [nu_dict["nu_0"][:, None], nu_dict["nu_1"][:, None]], axis=1,
    )

    # Get leptonic tau decay products and match them, then create leptonic taus from them
    gen_tau_neutrinos = higgs_family.tau_leptonic_decay_products[:, 0]
    for idx_1 in range(2):
        for idx_2 in range(2):
            nu_dict[f"nu_tau_{idx_1}_mask_{idx_2}"] = nu_dict[f"nu_{idx_1}"].deltaR(gen_tau_neutrinos[:, idx_2]) < 0.4
            # print(len(ak.ravel(ak.drop_none(ak.mask(nu_dict[f"nu_tau_{idx_1}_mask_{idx_2}"],
            # nu_dict[f"nu_tau_{idx_1}_mask_{idx_2}"] == True)))))
    for idx in range(2):
        nu_dict[f"tau_{idx}_mask"] = ak.concatenate(
            [nu_dict[f"nu_tau_0_mask_{idx}"][:, None], nu_dict[f"nu_tau_1_mask_{idx}"][:, None]], axis=1,
        )
        nu_dict[f"matched_tau_{idx}"] = ak.mask(tau_neutrinos, nu_dict[f"tau_{idx}_mask"])
        # ak.any(nu_dict["tau_1_mask"], axis=1)[:, None]], axis=1), axis=1)
    del nu_dict

    # Create mask for tau and tau neutrino matching
    gen_taus = higgs_family.taus
    # Get leptonic tau decay products
    muons = events.Muon
    muons = ak.mask(muons, ch_id_mask)       # for linting :|
    electrons = events.Electron
    electrons = ak.mask(electrons, ch_id_mask)
    gen_muons = higgs_family.tau_leptonic_decay_products[:, 1]
    gen_muons = gen_muons
    gen_electrons = higgs_family.tau_leptonic_decay_products[:, 2]
    gen_electrons = gen_electrons
    lep_decay_dict = {}
    gen_tau_neutrinos["charge"] = 0
    for lep in ["muons", "electrons"]:
        for idx in range(2):
            lep_decay_dict[f"{lep}_{idx}_mask"] = eval(lep).deltaR(eval(f"gen_{lep}[:, {idx}]")) < 0.4
            lep_decay_dict[f"{lep}_{idx}"] = ak.mask(eval(lep), lep_decay_dict[f"{lep}_{idx}_mask"])
            # remove events where more than 1 lep is matched
            lep_decay_dict[f"{lep}_{idx}"] = ak.drop_none(lep_decay_dict[f"{lep}_{idx}"], axis=1)
            lep_decay_dict[f"{lep}_{idx}_mask"] = ak.num(lep_decay_dict[f"{lep}_{idx}"], axis=1) > 1
            lep_decay_dict[f"{lep}_{idx}"] = ak.mask(
                lep_decay_dict[f"{lep}_{idx}"], lep_decay_dict[f"{lep}_{idx}_mask"], valid_when=False,
            )
            lep_decay_dict[f"{lep}_{idx}"] = ak.pad_none(lep_decay_dict[f"{lep}_{idx}"], 1, axis=1)
            # lep_decay_dict[f"{lep}_{idx}"] = lep_decay_dict[f"{lep}_{idx}"].add(gen_tau_neutrinos[:, idx])
    # Create awkward object to be able to attach_coffea_behavior
    tau_lept = ak.zip({
        "tau_lept_0": ak.pad_none(
            ak.concatenate([lep_decay_dict["muons_0"], lep_decay_dict["electrons_0"]], axis=1),
            2,
            axis=1),
        "tau_lept_1": ak.pad_none(
            ak.concatenate([lep_decay_dict["muons_1"], lep_decay_dict["electrons_1"]], axis=1),
            2,
            axis=1),
    }, with_name="tau_lept", depth_limit=1)
    tau_lept = attach_coffea_behavior_fn(tau_lept, collections={
        "tau_lept_0": {"type_name": "LorentzVector"},
        "tau_lept_1": {"type_name": "LorentzVector"},
    })
    if ak.sort(ak.num(ak.drop_none(tau_lept.tau_lept_0.x)), axis=-1, ascending=False)[0] | ak.sort(
            ak.num(ak.drop_none(tau_lept.tau_lept_1.x)), axis=-1, ascending=False)[0] > 1:
        raise ValueError("2 tau_0 or tau_1 leps")
    # shape arrays to remove empty entries
    tau_lept = ak.with_field(
        tau_lept,
        tau_lept.tau_lept_0[ak.argsort(ak.fill_none(tau_lept.tau_lept_0.x, EMPTY_FLOAT),
        ascending=False)][:, 0],
        "tau_lept_0",
    )
    tau_lept = ak.with_field(
        tau_lept,
        tau_lept.tau_lept_1[ak.argsort(ak.fill_none(tau_lept.tau_lept_1.x, EMPTY_FLOAT),
        ascending=False)][:, 0],
        "tau_lept_1",
    )
    tau_lept = attach_coffea_behavior_fn(tau_lept, collections={
        "tau_lept_0": {"type_name": "LorentzVector"},
        "tau_lept_1": {"type_name": "LorentzVector"},
    })
    # add dummy vectors for later sync with hadr taus
    dummy_tau = ak.zip(
        {
            "charge": EMPTY_FLOAT,
            "eta": EMPTY_FLOAT,
            "mass": EMPTY_FLOAT,
            "phi": EMPTY_FLOAT,
            "pt": EMPTY_FLOAT,
        },
    )
    tau_lept = ak.with_field(
        tau_lept,
        ak.fill_none(
            ak.mask(
                tau_lept.tau_lept_0, ak.fill_none(tau_lept.tau_lept_0.x, EMPTY_FLOAT) == EMPTY_FLOAT, valid_when=False,
            ), dummy_tau,
        ),
        "tau_lept_0",
    )
    tau_lept = ak.with_field(
        tau_lept,
        ak.fill_none(
            ak.mask(
                tau_lept.tau_lept_1, ak.fill_none(tau_lept.tau_lept_1.x, EMPTY_FLOAT) == EMPTY_FLOAT, valid_when=False,
            ), dummy_tau,
        ),
        "tau_lept_1",
    )
    # attach coffea behavior
    tau_lept = attach_coffea_behavior_fn(tau_lept, collections={
        "tau_lept_0": {"type_name": "LorentzVector"},
        "tau_lept_1": {"type_name": "LorentzVector"},
    })

    # free up some memory (maybe?)
    del lep_decay_dict

    # Maybe something like this would be easier if I could get it to work :s
    # tau_lept_0_mask = ak.fill_none(tau_lept.tau_lept_0.x, EMPTY_FLOAT) != EMPTY_FLOAT
    # tau_0 = ak.where(tau_lept_0_mask, tau_lept.tau_lept_0, tau_hadr_0)
    # tau_hadr_0_mask = ak.fill_none(tau_hadr_0.rho, EMPTY_FLOAT) != EMPTY_FLOAT

    # get hadronic taus and match them
    tau_jets = events.Tau
    tau_jets = ak.mask(tau_jets, ch_id_mask)
    tau_hadr_0_mask = tau_jets.deltaR(gen_taus[:, 0]) < 0.4
    tau_hadr_1_mask = tau_jets.deltaR(gen_taus[:, 1]) < 0.4
    tau_hadr_0 = ak.mask(tau_jets, tau_hadr_0_mask)
    tau_hadr_1 = ak.mask(tau_jets, tau_hadr_1_mask)
    # skip events where one tau jet is matched to two gen taus:
    tau_hadr_0 = ak.mask(tau_hadr_0, ak.num(ak.drop_none(tau_hadr_0.charge, axis=1), axis=1) == 2, valid_when=False)
    tau_hadr_1 = ak.mask(tau_hadr_1, ak.num(ak.drop_none(tau_hadr_1.charge, axis=1), axis=1) == 2, valid_when=False)
    if ak.sort(ak.num(ak.drop_none(tau_hadr_0.charge), axis=-1), ascending=False)[0] | ak.sort(
            ak.num(ak.drop_none(tau_hadr_1.charge), axis=-1), ascending=False)[0] > 1:
        raise ValueError("too many tau_hadr_0 or tau_hadr_1")
        # from IPython import embed
        # embed(header="too many taus in pdf_input_vars_reco_higgs")

    # shape arrays
    tau_hadr_0 = tau_hadr_0[ak.argsort(ak.fill_none(tau_hadr_0.charge, EMPTY_FLOAT), ascending=False)][:, 0]
    tau_hadr_1 = tau_hadr_1[ak.argsort(ak.fill_none(tau_hadr_1.charge, EMPTY_FLOAT), ascending=False)][:, 0]

    # tau_hadr_0 = tau_hadr_0.add(gen_tau_neutrinos[:, 0])
    # tau_hadr_1 = tau_hadr_1.add(gen_tau_neutrinos[:, 1])
    # add dummy vectors, steps below for concatenation with leptonic taus
    tau_hadr_0 = ak.mask(tau_hadr_0, ak.fill_none(tau_hadr_0.x, EMPTY_FLOAT) == EMPTY_FLOAT, valid_when=False)
    tau_hadr_1 = ak.mask(tau_hadr_1, ak.fill_none(tau_hadr_1.x, EMPTY_FLOAT) == EMPTY_FLOAT, valid_when=False)
    tau_hadr_0 = ak.fill_none(tau_hadr_0, dummy_tau)
    tau_hadr_1 = ak.fill_none(tau_hadr_1, dummy_tau)
    tau_hadr = ak.zip({
        "tau_hadr_0": tau_hadr_0,
        "tau_hadr_1": tau_hadr_1,
    }, depth_limit=1)
    tau_hadr = attach_coffea_behavior_fn(tau_hadr, collections={
        "tau_hadr_0": {"type_name": "LorentzVector"},
        "tau_hadr_1": {"type_name": "LorentzVector"},
    })

    # concatenate leptonic and hadronic taus
    tau_0 = ak.concatenate([tau_lept.tau_lept_0[:, None], tau_hadr.tau_hadr_0[:, None]], axis=1)
    tau_1 = ak.concatenate([tau_lept.tau_lept_1[:, None], tau_hadr.tau_hadr_1[:, None]], axis=1)
    # remove dummy objects
    # 1e4 results from transformation of dummy_vec_rho to pxpypzenergy coordinates
    # all in all very ugly TODO: is it physically ok to remove values below 1e4?
    tau_0 = ak.mask(tau_0, abs(tau_0.x) < 1e4)
    tau_1 = ak.mask(tau_1, abs(tau_1.x) < 1e4)
    tau_0 = tau_0[ak.argsort(ak.fill_none(tau_0.x, EMPTY_FLOAT), ascending=False)][:, 0]
    tau_1 = tau_1[ak.argsort(ak.fill_none(tau_1.x, EMPTY_FLOAT), ascending=False)][:, 0]
    del tau_hadr
    del tau_lept
    taus = ak.concatenate([tau_0[:, None], tau_1[:, None]], axis=1)
    del tau_0
    del tau_1

    matched_event_mask = ak.all(ak.fill_none(taus.x, EMPTY_FLOAT) != EMPTY_FLOAT, axis=1)

    # matched_hadr_taus = ak.concatenate([tau_hadr_0[:, None], tau_hadr_1[:, None]], axis=1)
    # matched_hadr_taus = ak.pad_none(matched_hadr_taus, 2, axis=1, clip=True)

    # Create tau array
    # matched_event_mask = ak.all(ak.concatenate([matched_nu_tau_event_mask[:, None], ak.all(ak.fill_none(
    #     matched_taus.eta, EMPTY_FLOAT) != EMPTY_FLOAT, axis=1)[:, None]], axis=1), axis=1)
    # print(ak.num(matched_event_mask[matched_event_mask == True], axis=0))
    # for ak.mask to work correctly, change shape of matched_event_mask:
    # matched_event_mask = ak.concatenate([matched_event_mask[:, None], matched_event_mask[:, None]], axis=1)

    # Match reco bjets to gen b's: Only fully matched events atm
    b_jets = events.HHBJet
    gen_bs = higgs_family.bottoms
    b_jet_0_mask = b_jets.deltaR(gen_bs[:, 0]) < 0.4
    b_jet_1_mask = b_jets.deltaR(gen_bs[:, 1]) < 0.4
    b_jet_0 = b_jets[b_jet_0_mask]
    b_jet_1 = b_jets[b_jet_1_mask]
    matched_b_jets = ak.concatenate([b_jet_0, b_jet_1], axis=1)
    matched_b_jets = ak.pad_none(matched_b_jets, 2, axis=1, clip=True)
    # Take only fully matched b events
    fully_matched_events_mask = ak.num(matched_b_jets, axis=1) == 2
    # Add to that the requirement of fully matched tau events
    fully_matched_events_mask = ak.all(
        ak.concatenate(
            [matched_event_mask[:, None], fully_matched_events_mask[:, None]], axis=1,
        ), axis=1,
    )
    # for ak.mask as above
    fully_matched_events_mask = ak.concatenate(
        [fully_matched_events_mask[:, None], fully_matched_events_mask[:, None]], axis=1,
    )
    matched_b_jets = ak.mask(matched_b_jets, fully_matched_events_mask)

    # Match taus and tau neutrinos with fully_matched_events_mask
    # add neutrinos (gen for now)

    # Build objects relevant to create input variables
    # #TODO: change neutrinos to reco
    # Build H_tautau: Add tau neutrinos to taus and then sum them event-wise
    taus = ak.mask(taus, fully_matched_events_mask)
    H_tautau = taus[:, 0].add(gen_tau_neutrinos[:, 0]).add(taus[:, 1].add(gen_tau_neutrinos[:, 1]))
    H_bb = matched_b_jets.sum(axis=1)
    h1 = H_bb
    b1 = matched_b_jets[:, 0]
    h2 = H_tautau
    tau1 = taus[:, 0]

    # Compute the pdf inputs
    dihiggs_system = H_bb.add(H_tautau)
    dihiggs_mass = dihiggs_system.mass
    dihiggs_system_pt = dihiggs_system.pt
    dihiggs_system_pz = dihiggs_system.pz
    dihiggs_system_phi = dihiggs_system.phi

    h1_cms_dihiggs = h1.boostCM_of(dihiggs_system)
    b1_cms_h1 = b1.boostCM_of(h1_cms_dihiggs.boostvec)
    # angle between b1 in cms of h1 and h1 in lab system
    cos_theta_cms_h1_b1 = signed_cos_deltaangle(b1_cms_h1, h1_cms_dihiggs)
    phi_cms_h1_b1 = b1_cms_h1.phi   # phi of b1 in h1's cms
    # boost h1 into cms of dihiggs system
    # angle between h1 in dihiggs cms and dihiggs in lab system
    cos_theta_h1 = signed_cos_deltaangle(h1_cms_dihiggs, dihiggs_system)
    phi_h1 = h1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h2 = h2.boostCM_of(dihiggs_system)
    tau1_cms_h2 = tau1.boostCM_of(h2.boostvec)
    # theta_cms_h2_tau1 = h2.deltaangle(tau1_cms_h2)   # angle between tau1 in cms of h2 and h2 in lab system
    cos_theta_cms_h2_tau1 = signed_cos_deltaangle(tau1_cms_h2, h2)
    phi_cms_h2_tau1 = tau1_cms_h2.phi   # phi of tau1 in h2's cms

    pdf_input_vars = ak.zip(
        {
            "dihiggs_mass": dihiggs_mass,
            "dihiggs_system_pt": dihiggs_system_pt,
            "dihiggs_system_pz": dihiggs_system_pz,
            "dihiggs_system_phi": dihiggs_system_phi,
            "cos_theta_h1": cos_theta_h1,
            "phi_h1": phi_h1,
            "cos_theta_cms_h2_tau1": cos_theta_cms_h2_tau1,
            "phi_cms_h2_tau1": phi_cms_h2_tau1,
            "cos_theta_cms_h1_b1": cos_theta_cms_h1_b1,
            "phi_cms_h1_b1": phi_cms_h1_b1,
        },
        with_name="pdf_input_vars",
    )

    events = set_ak_column(events, "pdf_input_vars_reco_higgs", pdf_input_vars)

    return events


@producer(
    uses={"top_family.*", "HHBJet", attach_coffea_behavior},
    produces={"pdf_input_vars_reco_top.*"},
)
def create_pdf_input_vars_reco_top(
    self: Producer, events: ak.Array, **kwargs,
) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_reco_top" that contains the inputs for the background likelihood in the di-tau
    case.
    """

    # We consider only the di-leptonic case in which both leptons are taus for now,
    # and of those only the full hadronic decays

    events = attach_coffea_behavior_fn(
        events,
        collections={
            "HHBJet": {"type_name": "LorentzVector"},
            "Tau": {"type_name": "LorentzVector"},
        },
    )

    # Invariant mass of visible taus and bs
    full_hadr_mask = ak.num(events.Tau, axis=1) == 2
    mtauvtauvbb = events.HHBJet[full_hadr_mask].sum(axis=1).add(events.Tau[full_hadr_mask].sum(axis=1))
    mtauvtauvbb = mtauvtauvbb.absolute()

    # Rapidity y of tau_vis and b system for both tops
