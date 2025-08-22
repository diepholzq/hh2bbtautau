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
    from IPython import embed
    embed(header="create_pdf_input_vars_higgs")
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
    theta_h1 = dihiggs_system.deltaangle(h1_cms_dihiggs)       # angle between h1 in dihiggs cms and dihiggs in
    # lab system. always positive, do we want that?
    cos_theta_h1 = np.cos(theta_h1)
    phi_h1 = h1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h2 = higgs_family.higgs[:, 1]
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
    input_vars_dict["theta_t"] = top.theta

    for i in range(2):
        input_vars_dict["W_t" + str(i) + "_phi"] = top_family[:, 2, i].boostCM_of(top_family[:, 0, i].boostvec).phi
        input_vars_dict["W_t" + str(i) + "_theta"] = top_family[:, 2, i].boostCM_of(top_family[:, 0, i].boostvec).theta

    # want child of W that is no neutrino: Either q qbar pair or lepton (11,13,15)
    # Which q to take?
    q_or_antilep_t0 = top_family[:, [3, 6], 0]
    q_or_antilep_t0_mask = q_or_antilep_t0.pdgId != EMPTY_FLOAT
    W_t0_child = q_or_antilep_t0[q_or_antilep_t0_mask].boostCM_of(top_family[:, 2, 0].boostvec)
    q_or_antilep_t1 = top_family[:, [3, 5], 1]
    q_or_antilep_t1_mask = q_or_antilep_t1.pdgId != EMPTY_FLOAT
    W_t1_child = q_or_antilep_t1[q_or_antilep_t1_mask].boostCM_of(top_family[:, 2, 1].boostvec)
    for i in range(2):
        input_vars_dict["W_t" + str(i) + "_child_phi"] = eval(f"W_t{i}_child").phi
        input_vars_dict["W_t" + str(i) + "_child_theta"] = eval(f"W_t{i}_child").theta

    pdf_input_vars = ak.zip(input_vars_dict, with_name="pdf_input_vars")
    events = set_ak_column(events, "pdf_input_vars", pdf_input_vars)
    W_t0_child = W_t0_child
    W_t1_child = W_t1_child
    return events
