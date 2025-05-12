# coding: utf-8

from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
# import vector

ak = maybe_import("awkward")


@producer(
    uses={"higgs_family.*", attach_coffea_behavior},
    produces={"pdf_input_vars.*"},
)
def create_pdf_input_vars(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
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
    theta_star_h_1 = h_1.deltaangle(h_1_cms_dihiggs)       # always positive, do we want that?
    phi_star_h_1 = h_1_cms_dihiggs.phi

    # boost tau into cms of htautau
    h_2 = higgs_family[:, 0, 1]
    tau_1 = higgs_family[:, 2, 0]
    tau_1_cms_h_2 = tau_1.boostCM_of(h_2.boostvec)
    theta_cms_h_2_tau_1 = tau_1.deltaangle(tau_1_cms_h_2)   # angle between tau_1 in lab system and h2 in its cms
    phi_cms_h_2_tau_1 = tau_1_cms_h_2.phi   # phi of tau_1 in h2's cms
    # theta_star_tau_tau = higgs_family[:,2,1].boostCM_of(h_2.boostvec).deltaangle(tau_1_cms_h_2)   # angle between
    # tau_1 and tau_2 in h2's cms

    # boost b1 into cms of hbb
    b_1 = higgs_family[:, 1, 0]
    b_1_cms_h_1 = b_1.boostCM_of(h_2.boostvec)
    theta_cms_h_1_b_1 = b_1.deltaangle(b_1_cms_h_1)     # angle between b_1 in lab system and h1 in its cms
    phi_cms_h_1_b_1 = b_1_cms_h_1.phi   # phi of b_1 in h1's cms

    pdf_input_vars = ak.zip({"dihiggs_mass": dihiggs_mass,
                             "dihiggs_system_pt": dihiggs_system_pt,
                             "dihiggs_system_pz": dihiggs_system_pz,
                             "dihiggs_system_phi": dihiggs_system_phi,
                             "theta_star_h_1": theta_star_h_1,
                             "phi_star_h_1": phi_star_h_1,
                             "theta_cms_h_2_tau_1": theta_cms_h_2_tau_1,
                             "phi_cms_h_2_tau_1": phi_cms_h_2_tau_1,
                             "theta_cms_h_1_b_1": theta_cms_h_1_b_1,
                             "phi_cms_h_1_b_1": phi_cms_h_1_b_1}, with_name="pdf_input_vars")

    events = set_ak_column(events, "pdf_input_vars", pdf_input_vars)
    return events
