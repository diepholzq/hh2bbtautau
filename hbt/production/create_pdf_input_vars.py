# coding: utf-8

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
import numpy as np
# import vector

ak = maybe_import("awkward")


def signed_cos_deltaangle(ax, ay, az, bx, by, bz):
    """
    Returns the signed angle between two vectors a and b.
    """
    a_times_b = ax * bx + ay * by + az * bz
    abs_a = np.sqrt(ax**2 + ay**2 + az**2)
    abs_b = np.sqrt(bx**2 + by**2 + bz**2)
    return a_times_b / (abs_a * abs_b)


@producer(
    uses={"higgs_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars.*"},
)
def create_pdf_input_vars_higgs(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new column "pdf_input_vars" that stores the PDF input variables for the Higgs decay products.
    """

    # Get higgs_family column and attach coffea behavior
    events = self[attach_coffea_behavior](events, collections={"higgs_family": {"type_name": "GenParticle",
        "check_attr": "metric_table", "skip_fields": "*Idx*G"}}, **kwargs)
    higgs_family = events.higgs_family

    # Define input vars
    dihiggs_system = higgs_family[:, 0].sum(axis=1)
    dihiggs_mass = dihiggs_system.mass
    dihiggs_system_pt = dihiggs_system.pt
    dihiggs_system_pz = dihiggs_system.pz
    dihiggs_system_phi = dihiggs_system.phi

    # boost h1 into cms of dihiggs system
    h_1 = higgs_family[:, 0, 0]
    h_1_cms_dihiggs = h_1.boostCM_of(dihiggs_system)
    theta_h_1 = dihiggs_system.deltaangle(h_1_cms_dihiggs)       # angle between h1 in dihiggs cms and dihiggs in
    # lab system. always positive, do we want that?
    cos_theta_h_1 = np.cos(theta_h_1)
    phi_star_h_1 = h_1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h_2 = higgs_family[:, 0, 1]
    tau_1 = higgs_family[:, 2, 0]
    tau_1_cms_h_2 = tau_1.boostCM_of(h_2.boostvec)
    theta_cms_h_2_tau_1 = h_2.deltaangle(tau_1_cms_h_2)   # angle between tau_1 in cms of h2 and h2 in lab system
    cos_theta_cms_h_2_tau_1 = np.cos(theta_cms_h_2_tau_1)
    phi_cms_h_2_tau_1 = tau_1_cms_h_2.phi   # phi of tau_1 in h2's cms
    # theta_star_tau_tau = higgs_family[:,2,1].boostCM_of(h_2.boostvec).deltaangle(tau_1_cms_h_2)   # angle between
    # tau_1 and tau_2 in h2's cms

    # boost b1 into cms of hbb
    b_1 = higgs_family[:, 1, 0]
    b_1_cms_h_1 = b_1.boostCM_of(h_1.boostvec)
    theta_cms_h_1_b_1 = h_1.deltaangle(b_1_cms_h_1)     # angle between b_1 in cms of h1 and h1 in lab system
    cos_theta_cms_h_1_b_1 = np.cos(theta_cms_h_1_b_1)
    phi_cms_h_1_b_1 = b_1_cms_h_1.phi   # phi of b_1 in h1's cms

    pdf_input_vars = ak.zip({"dihiggs_mass": dihiggs_mass,
                             "dihiggs_system_pt": dihiggs_system_pt,
                             "dihiggs_system_pz": dihiggs_system_pz,
                             "dihiggs_system_phi": dihiggs_system_phi,
                             "cos_theta_h_1": cos_theta_h_1,
                             "phi_star_h_1": phi_star_h_1,
                             "cos_theta_cms_h_2_tau_1": cos_theta_cms_h_2_tau_1,
                             "phi_cms_h_2_tau_1": phi_cms_h_2_tau_1,
                             "cos_theta_cms_h_1_b_1": cos_theta_cms_h_1_b_1,
                             "phi_cms_h_1_b_1": phi_cms_h_1_b_1}, with_name="pdf_input_vars")

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

    # Get higgs_family column and attach coffea behavior
    events = self[attach_coffea_behavior](events, collections={"top_family": {"type_name": "GenParticle",
        "check_attr": "metric_table", "skip_fields": "*Idx*G"}}, **kwargs)
    top_family = events.top_family

    # Define input vars
    ttbar_system = top_family[:, 0].sum(axis=1)
    ttbar_mass = ttbar_system.mass
    ttbar_system_pt = ttbar_system.pt
    ttbar_system_pz = ttbar_system.pz
    ttbar_system_phi = ttbar_system.phi

    # boost t1 into cms of ttbar system
    t_1 = top_family[:, 0, 0]
    t_1_cms_ttbar = t_1.boostCM_of(ttbar_system)
    theta_t_1 = ttbar_system.deltaangle(t_1_cms_ttbar)       # always positive, do we want that?
    cos_theta_t_1 = np.cos(theta_t_1)
    phi_star_t_1 = t_1_cms_ttbar.phi
    # boost b1 into cms of t1
    b_1 = top_family[:, 1, 0]
    b_1_cms_t_1 = b_1.boostCM_of(t_1.boostvec)
    theta_cms_t_1_b_1 = t_1.deltaangle(b_1_cms_t_1)     # angle between b_1 in cms of t1 and t_1 in lab system
    cos_theta_cms_t1_b_1 = np.cos(theta_cms_t_1_b_1)
    phi_cms_t_1_b_1 = b_1_cms_t_1.phi   # phi of b_1 in t1's cms
    theta_cms_t_1_b_1_W = b_1_cms_t_1.deltaangle(top_family[:, 2, 0].boostCM_of(t_1.boostvec))  # angle between b_1 in
    # cms of t1 and W in t1's cms
    cos_theta_cms_t_1_b_1_W = np.cos(theta_cms_t_1_b_1_W)
    # from IPython import embed; embed(header="debugging pdf_inputs")
    tau_event_mask = ak.any(top_family[:, 6].pdgId == -15, axis=1)
    tau_events = ak.mask(top_family, tau_event_mask)
    tau_nu_cms_t1 = ak.flatten(tau_events[:, [6, 7]], axis=2)[abs(ak.flatten(
        tau_events[:, [6, 7]], axis=2).pdgId) < 99999].sum(axis=1).boostCM_of(t_1.boostvec)

    antitaus = tau_events[:, 6]
    antitaus = antitaus[abs(antitaus.pdgId) < 99999]
    theta_cms_t_1_tau_nu_tau_b_1 = tau_nu_cms_t1.deltaangle(b_1_cms_t_1)
    theta_cms_t_1_tau_b_1 = antitaus.boostCM_of(t_1.boostvec).deltaangle(b_1_cms_t_1)

    pdf_input_vars = ak.zip({"ttbar_mass": ttbar_mass,
                             "ttbar_system_pt": ttbar_system_pt,
                             "ttbar_system_pz": ttbar_system_pz,
                             "ttbar_system_phi": ttbar_system_phi,
                             "cos_theta_t_1": cos_theta_t_1,
                             "phi_star_t_1": phi_star_t_1,
                             "cos_theta_cms_t_1_b_1": cos_theta_cms_t1_b_1,
                             "phi_cms_t_1_b_1": phi_cms_t_1_b_1,
                             "cos_theta_cms_t1_b1_W": cos_theta_cms_t_1_b_1_W,
                             "theta_cms_t_1_tau_nu_tau_b_1": theta_cms_t_1_tau_nu_tau_b_1,
                             "theta_cms_t_1_tau_b_1": theta_cms_t_1_tau_b_1,
                             }, with_name="pdf_input_vars", depth_limit=1)

    events = set_ak_column(events, "pdf_input_vars", pdf_input_vars)
    return events
