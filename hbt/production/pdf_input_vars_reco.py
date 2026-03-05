from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn
from hbt.production.create_pdf_input_vars import signed_cos_deltaangle
import vector

np = maybe_import("numpy")
ak = maybe_import("awkward")


def calculate_rapidity(particle):
    particle_y = 1 / 2 * np.log(
        (particle.energy + particle.pz) / (particle.energy - particle.pz))
    particle_y = np.nan_to_num(particle_y)
    return particle_y


def calculate_constraint_term(
        corrected_distr1: ak.Array,
        corrected_distr2: ak.Array,
        uncorr_distr1: ak.Array,
        uncorr_distr2: ak.Array,
        ch_id_mask: ak.Array,
) -> np.array:
    """Calculates log of Likelihood that indicates the probability of a measurement agreeing with the correction
    """
    mean_correction1 = np.mean(corrected_distr1[ch_id_mask] - uncorr_distr1[ch_id_mask])
    mean_correction2 = np.mean(corrected_distr2[ch_id_mask] - uncorr_distr2[ch_id_mask])
    sigma_correction1 = np.std(corrected_distr1[ch_id_mask] - uncorr_distr1[ch_id_mask])
    sigma_correction2 = np.std(corrected_distr2[ch_id_mask] - uncorr_distr2[ch_id_mask])
    constr_term = 1 / 2 * (((corrected_distr1 - uncorr_distr1 - mean_correction1) / (sigma_correction1))**2 +
                        ((corrected_distr2 - uncorr_distr2 - mean_correction2) / (sigma_correction2))**2)
    return constr_term


@producer(
    uses={"HHBJet", "channel_id", "Tau.*", attach_coffea_behavior},
    produces={"pdf_input_vars_reco_higgs.*"},
)
def create_pdf_input_vars_reco_higgs(
    self: Producer,
    events: ak.Array,
    **kwargs,
) -> ak.Array:
    """
    Creates a new column "pdf_input_vars_reco_higgs" that contains the inputs for the signal likelihood. Only events
    with channel_id 3, i.e. events classified as full hadronic tau decays, are evaluated.
    """
    ch_id_mask = events.channel_id == 3

    behaving_columns = attach_coffea_behavior_fn(
        events,
        collections={
            "HHBJet": {
                "type_name": "LorentzVector",
            },
            "Tau": {
                "type_name": "LorentzVector",
            },
        },
    )
    # Build h1 by adding the b jets
    b_jets = behaving_columns.HHBJet
    m_bb_rec = b_jets[:, 0].add(b_jets[:, 1]).mass

    m_bb = 125
    vector.register_awkward()
    b_corrected = vector.zip({
        "energy": b_jets.energy * m_bb / m_bb_rec,
        "px": b_jets.px * m_bb / m_bb_rec,
        "py": b_jets.py * m_bb / m_bb_rec,
        "pz": b_jets.pz * m_bb / m_bb_rec,
    })
    constr_term_b = calculate_constraint_term(b_corrected[:, 0].pt,
                                              b_corrected[:, 1].pt,
                                              b_jets[:, 0].pt,
                                              b_jets[:, 1].pt, ch_id_mask)
    # h1 = b_jets[:, 0].add(b_jets[:, 1])
    h1 = b_corrected[:, 0].add(b_corrected[:, 1])
    # h1 = ak.where(ch_id_mask, h1, dummy_vec)

    # Build h2 by adding the tau jets
    taus = behaving_columns.Tau
    taus = ak.mask(taus, ch_id_mask)
    dummy_tau = ak.drop_none(ak.firsts(events.Tau))[0]
    taus = ak.fill_none(taus, [dummy_tau, dummy_tau], axis=0)
    taus = ak.zip({"taus": taus})
    taus = attach_coffea_behavior_fn(
        taus,
        collections={
            "taus": {
                "type_name": "LorentzVector",
            },
        },
    )
    m_tautau_rec = taus.taus[:, 0].add(taus.taus[:, 1]).mass
    corr_factor_tau = 125 / m_tautau_rec
    taus_corr = vector.zip({
        "energy": taus.taus.energy * corr_factor_tau,
        "px": taus.taus.px * corr_factor_tau,
        "py": taus.taus.py * corr_factor_tau,
        "pz": taus.taus.pz * corr_factor_tau,
        "charge": taus.taus.charge,
    })
    taus_uncorr = vector.zip({
        "energy": taus.taus.energy,
        "px": taus.taus.px,
        "py": taus.taus.py,
        "pz": taus.taus.pz,
        "charge": taus.taus.charge,
    })
    h2 = taus_corr[:, 0].add(taus_corr[:, 1])
    # h2 = ak.where(ch_id_mask, taus_corr[:, 0].add(taus_corr[:, 1]), EMPTY_FLOAT)

    # Calculate b inputs and choose random b for each event
    rng = np.random.default_rng()
    which_b = rng.integers(0, 1, endpoint=True, size=len(events))
    b_cms_h1_opt1 = b_corrected[:, 0].boostCM_of(h1)
    b_cms_h1_opt2 = b_corrected[:, 1].boostCM_of(h1)
    cos_theta_cms_h1_b_opt1 = signed_cos_deltaangle(b_cms_h1_opt1, h1)
    cos_theta_cms_h1_b_opt2 = signed_cos_deltaangle(b_cms_h1_opt2, h1)
    phi_cms_h1_b_opt1 = b_cms_h1_opt1.phi
    phi_cms_h1_b_opt2 = b_cms_h1_opt2.phi
    cos_theta_cms_h1_b1 = ak.where(which_b == 0, cos_theta_cms_h1_b_opt1,
                                   cos_theta_cms_h1_b_opt2)
    phi_cms_h1_b1 = ak.where(which_b == 0, phi_cms_h1_b_opt1,
                             phi_cms_h1_b_opt2)

    # Calculate tau inputs
    tau_charge_mask = ak.argsort(taus_corr.charge, axis=1, ascending=False)
    taus_corr_sorted = taus_corr[tau_charge_mask]
    constr_term_tau = calculate_constraint_term(
        taus_corr[:, 0].pt,
        taus_corr[:, 1].pt,
        taus_uncorr[tau_charge_mask][:, 0].pt,
        taus_uncorr[tau_charge_mask][:, 1].pt,
        ch_id_mask,
    )
    tau_vis1 = taus_corr_sorted[:, 0]
    tau_vis1_cms_h2 = tau_vis1.boostCM_of(h2)
    cos_theta_cms_h2_tau_vis1 = signed_cos_deltaangle(tau_vis1_cms_h2, h2)
    phi_cms_h2_tau_vis1 = tau_vis1_cms_h2.phi

    # Calculate dihiggs inputs
    dihiggs_system = h1.add(h2)
    dihiggs_mass = dihiggs_system.mass
    dihiggs_system_pt = dihiggs_system.pt
    dihiggs_system_pz = dihiggs_system.pz
    dihiggs_system_phi = dihiggs_system.phi

    # Calculate h1 inputs
    h1_cms_dihiggs = h1.boostCM_of(dihiggs_system)
    cos_theta_h1 = signed_cos_deltaangle(h1_cms_dihiggs, dihiggs_system)
    phi_h1 = h1_cms_dihiggs.phi

    # Create the column
    pdf_input_vars_reco_higgs = ak.zip(
        {
            "dihiggs_mass": dihiggs_mass,
            "dihiggs_system_pt": dihiggs_system_pt,
            "dihiggs_system_pz": dihiggs_system_pz,
            "dihiggs_system_phi": dihiggs_system_phi,
            "cos_theta_h1": cos_theta_h1,
            "phi_h1": phi_h1,
            "cos_theta_cms_h2_tau_vis1": cos_theta_cms_h2_tau_vis1,
            "phi_cms_h2_tau_vis1": phi_cms_h2_tau_vis1,
            "cos_theta_cms_h1_b1": cos_theta_cms_h1_b1,
            "phi_cms_h1_b1": phi_cms_h1_b1,
            "constr_term_tau": constr_term_tau,
            "constr_term_b": constr_term_b,
        },
        with_name="pdf_input_vars_reco_higgs")
    pdf_input_vars_reco_higgs = ak.mask(pdf_input_vars_reco_higgs, ch_id_mask)
    events = set_ak_column(events, "pdf_input_vars_reco_higgs", pdf_input_vars_reco_higgs)

    return events


@producer(
    uses={"Tau", "HHBJet", attach_coffea_behavior},
    produces={"pdf_input_vars_reco_top.*"},
)
def create_pdf_input_vars_reco_top(
    self: Producer,
    events: ak.Array,
    **kwargs,
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
            "HHBJet": {
                "type_name": "LorentzVector",
            },
            "Tau": {
                "type_name": "LorentzVector",
            },
        },
    )

    # Invariant mass of visible taus and bs
    # full_hadr_mask = ak.num(events.Tau, axis=1) == 2

    # tau_vis tau_vis b b system (t_vis tbar_vis)
    # tt_vis_system = events.HHBJet[full_hadr_mask].sum(axis=1).add(
    #     events.Tau[full_hadr_mask].sum(axis=1))
    # tt_vis_system = events.HHBJet.sum(axis=1).add(
    #     events.Tau.sum(axis=1))
    #
    # tt_vis_system_mass = tt_vis_system.absolute()
    # tt_vis_system_pt = tt_vis_system.pt
    # tt_vis_system_pz = tt_vis_system.pz
    # tt_vis_system_phi = tt_vis_system.phi

    # t_1_vis system
    tau_charge_mask = ak.argsort(events.Tau.charge, axis=1, ascending=False)
    taus_sorted = events.Tau[tau_charge_mask]
    # taus_sorted = taus_sorted[full_hadr_mask]
    rng = np.random.default_rng()
    which_b = rng.integers(0, 1, endpoint=True, size=len(events))
    b1_random = ak.where(which_b == 0, events.HHBJet[:, 0], events.HHBJet[:, 1])
    # b1_random = b1_random[full_hadr_mask]
    b2_random = ak.where(which_b == 0, events.HHBJet[:, 1], events.HHBJet[:, 0])
    # b2_random = b2_random[full_hadr_mask]
    taus_sorted = ak.pad_none(taus_sorted, 2, axis=1, clip=True)
    t1_vis = taus_sorted[:, 0].add(b1_random)
    t2_vis = taus_sorted[:, 0].add(b2_random)

    tt_vis_system = t1_vis.add(t2_vis)
    tt_vis_system_mass = tt_vis_system.absolute()
    tt_vis_system_pt = tt_vis_system.pt
    tt_vis_system_pz = tt_vis_system.pz
    tt_vis_system_phi = tt_vis_system.phi

    t1_vis = t1_vis.boostCM_of(tt_vis_system)
    t2_vis = t2_vis.boostCM_of(tt_vis_system)
    t_vis_y_diff = calculate_rapidity(t1_vis) - calculate_rapidity(t2_vis)
    t1_vis_phi = t1_vis.phi

    # tau_vis b system (the same as t_1_vis ??)
    M_lb = t1_vis.absolute()
    cos_theta_star = 2 * M_lb**2 / (175**2 - 80.3**2 - 1.7**2) - 1
    lb_y = calculate_rapidity(t1_vis)

    pdf_input_vars_reco_top = ak.zip({
        "tt_vis_system_mass": tt_vis_system_mass,
        "tt_vis_system_pt": tt_vis_system_pt,
        "tt_vis_system_pz": tt_vis_system_pz,
        "tt_vis_system_phi": tt_vis_system_phi,
        "t_vis_y_diff": t_vis_y_diff,
        "t1_vis_phi": t1_vis_phi,
        "cos_theta_star": cos_theta_star,
        "lb_y": lb_y,
    }, with_name="pdf_input_vars_reco_top")
    pdf_input_vars_reco_top = ak.mask(pdf_input_vars_reco_top, events.channel_id == 3)
    events = set_ak_column(events, "pdf_input_vars_reco_top", pdf_input_vars_reco_top)

    return events


@producer(
    uses={
        create_pdf_input_vars_reco_top, create_pdf_input_vars_reco_higgs,
    },
    produces={
        create_pdf_input_vars_reco_top, create_pdf_input_vars_reco_higgs,
    },
)
def pdf_inputs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    events = self[create_pdf_input_vars_reco_higgs](events, **kwargs)
    events = self[create_pdf_input_vars_reco_top](events, **kwargs)
    return events
