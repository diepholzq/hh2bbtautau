from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import attach_coffea_behavior as attach_coffea_behavior_fn
from hbt.production.create_pdf_input_vars import signed_cos_deltaangle
import vector
import jax

np = maybe_import("numpy")
ak = maybe_import("awkward")


def calculate_rapidity(particle):
    particle_y = 1 / 2 * np.log(
        (particle.energy + particle.pz) / (particle.energy - particle.pz))
    particle_y = np.nan_to_num(particle_y)
    return particle_y


def signed_cos_deltaangle_for_jax(inputs):
    """
    Returns the signed angle between two vectors a and b.
    Args:
        inputs: Array of the thre-momenta of particles a and b, in order px, py, pz
                for particle a first, particle b second

    """
    ax, ay, az = inputs[0], inputs[1], inputs[2]
    bx, by, bz = inputs[3], inputs[4], inputs[5]
    a_times_b = ax * bx + ay * by + az * bz
    abs_a = np.sqrt(ax**2 + ay**2 + az**2)
    abs_b = np.sqrt(bx**2 + by**2 + bz**2)
    return a_times_b / (abs_a * abs_b)


def calculate_constraint_term(
    corrected_distr1,
        corrected_distr2,
        uncorr_distr1,
        uncorr_distr2,
        ch_id_mask,
) -> np.array:
    """Calculates log of Likelihood that indicates the probability of a measurement agreeing with the correction
    """
    mean_correction1 = np.mean(
        corrected_distr1[ch_id_mask] - uncorr_distr1[ch_id_mask])
    mean_correction2 = np.mean(
        corrected_distr2[ch_id_mask] - uncorr_distr2[ch_id_mask])
    sigma_correction1 = np.std(
        corrected_distr1[ch_id_mask] - uncorr_distr1[ch_id_mask])
    sigma_correction2 = np.std(
        corrected_distr2[ch_id_mask] - uncorr_distr2[ch_id_mask])
    constr_term = 1 / 2 * (((corrected_distr1 - uncorr_distr1 - mean_correction1) / (sigma_correction1))**2 +
                           ((corrected_distr2 - uncorr_distr2 - mean_correction2) / (sigma_correction2))**2)
    return constr_term


def return_deviations_where_necessary(m_1_sq, m_2_sq, both_false_mask):
    """Provided the mask that defines where both M_lb^2 are bigger than M_t^2 - M_W^2, this function returns
    for one b assignment hypothesis an array where the sum of the deviations to M_t^2 - M_W^2 is stored per event,
    provided that the event is selected by the above mask and that the deviations are positive, i.e. M_lb is larger
    than the constraining mass difference
    """
    m_tw_sq = 175**2 - 80.3**2
    is_bigger1 = ak.fill_none(m_1_sq, 0) > m_tw_sq
    m_tw_sq_arr_1 = np.ones_like(both_false_mask, dtype=float) * m_tw_sq
    m_tw_sq_arr_1 = ak.mask(m_tw_sq_arr_1, both_false_mask)
    m_tw_sq_arr_1 = ak.mask(m_tw_sq_arr_1, is_bigger1)
    m_tw_sq_arr_1 = ak.fill_none(m_tw_sq_arr_1, 0)
    m_1_sq_masked = ak.fill_none(ak.mask(m_1_sq, both_false_mask), 0)
    is_bigger2 = ak.fill_none(m_2_sq, 0) > m_tw_sq
    m_tw_sq_arr_2 = np.ones_like(both_false_mask, dtype=float) * m_tw_sq
    m_tw_sq_arr_2 = ak.mask(m_tw_sq_arr_2, both_false_mask)
    m_tw_sq_arr_2 = ak.mask(m_tw_sq_arr_2, is_bigger2)
    m_tw_sq_arr_2 = ak.fill_none(m_tw_sq_arr_2, 0)
    m_2_sq_masked = ak.fill_none(ak.mask(m_2_sq, both_false_mask), 0)
    deviations = (
        ak.fill_none(ak.mask(m_1_sq_masked, is_bigger1), 0) - m_tw_sq_arr_1 +
        ak.fill_none(ak.mask(m_2_sq_masked, is_bigger2), 0) - m_tw_sq_arr_2
    )
    return deviations


def det_coords_to_fourmomentum(inputs):
    """Transforms detector coordinates to four momentum space (px, py, pz, energy)
    Args:
        inputs: Array of pt, eta, phi, m in that order
    Returns:
        four_momentum: Array of px, py, pz, e in that order
    """
    pt, eta, phi, m = inputs[0], inputs[1], inputs[2], inputs[3]
    px = pt * jax.numpy.cos(phi)
    py = pt * jax.numpy.sin(phi)
    pz = pt * jax.numpy.sinh(eta)
    momentum = pt * jax.numpy.cosh(eta)
    energy = jax.numpy.sqrt(m**2 + momentum**2)
    return jax.numpy.stack([
        px, py, pz, energy,
    ], axis=1)


def fourmomentum_to_det_coord(inputs):
    """Transforms four momentum space coordinates (px, py, pz, energy) to detector space (pt, eta, phi, m)
    Args:
        inputs: Array of px, py, pz, energy in that order
    Returns:
        det_coords: Array of pt, eta, phi, m in that order
    """
    px, py, pz, energy = inputs[0], inputs[1], inputs[2], inputs[3]
    pt = jax.numpy.sqrt(px**2 + py**2)
    phi = jax.numpy.arctan2(py, px)
    eta = jax.numpy.arcsinh(pz / pt)
    m = jax.numpy.sqrt(energy**2 - (pt * jax.numpy.cosh(eta))**2)
    return jax.numpy.stack([
        pt, eta, phi, m,
    ], axis=1, dtype=np.float64)


def four_vec_sum(inputs):
    px1, py1, pz1, e1 = inputs[0], inputs[1], inputs[2], inputs[3]
    px2, py2, pz2, e2 = inputs[4], inputs[5], inputs[6], inputs[7]
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2
    return jax.numpy.stack([
        px, py, pz, e,
    ], axis=0).T


def dot_product(inputs):
    px1, py1, pz1 = inputs[0], inputs[1], inputs[2]
    px2, py2, pz2 = inputs[3], inputs[4], inputs[5]
    dot = px1 * px2 + py1 * py2 + pz1 * pz2
    return dot


def calculate_velocity(inputs):
    """ Calculates beta (v/c)
    """
    px, py, pz, energy = inputs[0], inputs[1], inputs[2], inputs[3]

    p_vec = jax.numpy.array([px, py, pz])

    # beta = momentum / energy
    beta = p_vec / energy

    return beta


def boost_a_cm_of_b(inputs, mass_b):
    # inputs: shape (8,)
    """Boosts particle a in to the rest frame of particle b
    Args:
        inputs: Array of the four-momenta of particles a and b in unboosted cms, in order px, py, pz, energy
                for particle a first, particle b second
        mass_b: Array with mass of particle b per event
    Returns:
        boosted_p: Boosted four momentum as jax.numpy.stack in above form (only for one particle obvsly)
    """
    a_px, a_py, a_pz, a_energy = inputs[0], inputs[1], inputs[2], inputs[3]
    b_energy = inputs[7]
    b_mass = mass_b

    p_vec_a = jax.numpy.array([a_px, a_py, a_pz])

    gamma_b = b_energy / b_mass

    # Calculate velocities of particles as seen in unboosted frame
    v_b = calculate_velocity(inputs[4:8])

    # Relative velocity is velocity of b in unboosted cms
    rel_v = v_b
    norm_v = jax.numpy.linalg.norm(rel_v, axis=0)
    normal_n = rel_v / norm_v

    # inputs_dot = jax.numpy.stack([inputs[0], inputs[1], inputs[2], normal_n[0], normal_n[1], normal_n[2]], axis=0)
    # dot_normal_n_p_vec_a = dot_product(inputs_dot)
    boosted_e = gamma_b * (a_energy - norm_v * jax.numpy.linalg.vecdot(p_vec_a, normal_n, axis=0))
    boosted_p = (
        p_vec_a + (gamma_b - 1) * (jax.numpy.linalg.vecdot(p_vec_a, normal_n, axis=0)) * normal_n - gamma_b * norm_v * a_energy * normal_n
    )
    # from IPython import embed
    # embed(header="boosting")
    return jax.numpy.stack([
        boosted_p[0], boosted_p[1], boosted_p[2], boosted_e,
    ], axis=0).T


def calculate_invariant_mass(inputs):
    """Calculates the invariant mass of two particles given their four-momenta
    Args:
        inputs: jax.numpy.stack of the four-momenta. Order: particle 1 px, py, pz, energy, then particle 2
    Returns:
        inv_mass: Invariant mass of the two particles
    """
    # inputs: shape (8,)
    part1_px, part1_py, part1_pz, part1_e = inputs[0], inputs[1], inputs[2], inputs[3]
    part2_px, part2_py, part2_pz, part2_e = inputs[4], inputs[5], inputs[6], inputs[7]

    px = part1_px + part2_px
    py = part1_py + part2_py
    pz = part1_pz + part2_pz
    e = part1_e + part2_e

    inv_mass = jax.numpy.sqrt(e**2 - (px**2 + py**2 + pz**2))
    return inv_mass


def calculate_dihiggs_mass(inputs):
    # inputs: shape (16,)
    b1_px, b1_py, b1_pz, b1_e = inputs[0], inputs[1], inputs[2], inputs[3]
    b2_px, b2_py, b2_pz, b2_e = inputs[4], inputs[5], inputs[6], inputs[7]
    tau1_px, tau1_py, tau1_pz, tau1_e = inputs[8], inputs[9], inputs[10], inputs[11]
    tau2_px, tau2_py, tau2_pz, tau2_e = inputs[12], inputs[13], inputs[14], inputs[15]

    h1_px = b1_px + b2_px
    h1_py = b1_py + b2_py
    h1_pz = b1_pz + b2_pz
    h1_e = b1_e + b2_e

    h2_px = tau1_px + tau2_px
    h2_py = tau1_py + tau2_py
    h2_pz = tau1_pz + tau2_pz
    h2_e = tau1_e + tau2_e

    dh_px = h1_px + h2_px
    dh_py = h1_py + h2_py
    dh_pz = h1_pz + h2_pz
    dh_e = h1_e + h2_e

    dh_mass = jax.numpy.sqrt(dh_e**2 - (dh_px**2 + dh_py**2 + dh_pz**2))
    return dh_mass


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
    ch_id_mask = ak.to_numpy(ch_id_mask)

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

    # get four-momenta of b's and taus
    # Select random b as b1, other as b2
    rng = np.random.default_rng()
    which_b1 = rng.integers(0, 1, endpoint=True, size=len(events))
    which_b2 = np.where(which_b1 == 0, 1, 0)
    b_mask = np.concatenate([which_b1[:, None], which_b2[:, None]], axis=1)
    b_mask = ak.Array([b_mask])[0]
    sorted_bs = events.HHBJet[b_mask]
    b_pt = ak.to_numpy(sorted_bs.pt, allow_missing=False)
    b_eta = ak.to_numpy(sorted_bs.eta, allow_missing=False)
    b_phi = ak.to_numpy(sorted_bs.phi, allow_missing=False)
    b_mass = ak.to_numpy(sorted_bs.mass, allow_missing=False)
    tau_charge_mask = ak.argsort(events.Tau.charge, axis=1, ascending=False)
    sorted_taus = events.Tau[tau_charge_mask]
    tau_pt = ak.to_numpy(
        ak.fill_none(ak.pad_none(sorted_taus.pt, 2, axis=1, clip=True), -99999.0), allow_missing=False,
    )
    tau_eta = ak.to_numpy(
        ak.fill_none(ak.pad_none(sorted_taus.eta, 2, axis=1, clip=True), -99999.0), allow_missing=False,
    )
    tau_phi = ak.to_numpy(
        ak.fill_none(ak.pad_none(sorted_taus.phi, 2, axis=1, clip=True), -99999.0), allow_missing=False,
    )
    tau_mass = ak.to_numpy(
        ak.fill_none(ak.pad_none(sorted_taus.mass, 2, axis=1, clip=True), -99999.0), allow_missing=False,
    )

    # Inputs for higgs input function
    b1_inputs = jax.numpy.stack([
        b_pt[:, 0], b_eta[:, 0], b_phi[:, 0], b_mass[:, 0],
    ], axis=0)
    b2_inputs = jax.numpy.stack([
        b_pt[:, 1], b_eta[:, 1], b_phi[:, 1], b_mass[:, 1],
    ], axis=0)
    tau1_inputs = jax.numpy.stack([
        tau_pt[:, 0], tau_eta[:, 0], tau_phi[:, 0], tau_mass[:, 0],
    ], axis=0)
    tau2_inputs = jax.numpy.stack([
        tau_pt[:, 1], tau_eta[:, 1], tau_phi[:, 1], tau_mass[:, 1],
    ], axis=0)

    # --------------------------------------------- Ab hier: part. Ableitungen relevant -----------------------
    b1 = det_coords_to_fourmomentum(b1_inputs)
    b2 = det_coords_to_fourmomentum(b2_inputs)
    tau1 = det_coords_to_fourmomentum(tau1_inputs)
    tau2 = det_coords_to_fourmomentum(tau2_inputs)

    # Calculate invariant mass of bb system for correction term
    # shape: 8,batch_size
    m_bb_rec = calculate_invariant_mass(
        jax.numpy.concatenate([b1, b2], axis=1).T)

    # Calculate correction terms: b
    m_bb = 125
    b1_corrected = jax.numpy.stack([
        b1[:, 0] * m_bb / m_bb_rec,
        b1[:, 1] * m_bb / m_bb_rec,
        b1[:, 2] * m_bb / m_bb_rec,
        b1[:, 3] * m_bb / m_bb_rec,

    ], axis=1)
    b1_corrected_detspace = fourmomentum_to_det_coord(b1_corrected.T)
    b2_corrected = jax.numpy.stack([
        b2[:, 0] * m_bb / m_bb_rec,
        b2[:, 1] * m_bb / m_bb_rec,
        b2[:, 2] * m_bb / m_bb_rec,
        b2[:, 3] * m_bb / m_bb_rec,

    ], axis=1)
    b2_corrected_detspace = fourmomentum_to_det_coord(b2_corrected.T)
    constr_term_b = calculate_constraint_term(
        b1_corrected_detspace[:, 0],
        b2_corrected_detspace[:, 0],
        b_pt[:, 0],
        b_pt[:, 1],
        ch_id_mask,
    )

    # Calculate constraint terms: taus
    m_tautau = 125
    m_tautau_rec = calculate_invariant_mass(
        jax.numpy.concatenate([tau1, tau2], axis=1).T)
    m_tautau_rec = jax.numpy.nan_to_num(m_tautau_rec, nan=-99999.)
    tau1_corrected = jax.numpy.stack([
        tau1[:, 0] * m_tautau / m_tautau_rec,
        tau1[:, 1] * m_tautau / m_tautau_rec,
        tau1[:, 2] * m_tautau / m_tautau_rec,
        tau1[:, 3] * m_tautau / m_tautau_rec,

    ], axis=1)
    tau1_corrected_detspace = fourmomentum_to_det_coord(tau1_corrected.T)
    tau2_corrected = jax.numpy.stack([
        tau2[:, 0] * m_tautau / m_tautau_rec,
        tau2[:, 1] * m_tautau / m_tautau_rec,
        tau2[:, 2] * m_tautau / m_tautau_rec,
        tau2[:, 3] * m_tautau / m_tautau_rec,

    ], axis=1)
    tau2_corrected_detspace = fourmomentum_to_det_coord(tau2_corrected.T)
    constr_term_tau = calculate_constraint_term(
        tau1_corrected_detspace[:, 0],
        tau2_corrected_detspace[:, 0],
        tau_pt[:, 0],
        tau_pt[:, 1],
        ch_id_mask,
    )

    # Build H_bb
    h1 = four_vec_sum(jax.numpy.concatenate([b1_corrected, b2_corrected], axis=1).T)
    h1_detspace = fourmomentum_to_det_coord(h1.T)

    # Build H_tautau
    h2 = four_vec_sum(jax.numpy.concatenate([tau1_corrected, tau2_corrected], axis=1).T)
    h2_detspace = fourmomentum_to_det_coord(h2.T)

    # Calculate b inputs
    b_cms_h1 = boost_a_cm_of_b(jax.numpy.concatenate([b1_corrected, h1], axis=1).T, h1_detspace[:, 3])
    b_cms_h1_detspace = fourmomentum_to_det_coord(b_cms_h1.T)
    cos_theta_cms_h1_b1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([b_cms_h1[:, :3], h1[:, :3]], axis=1).T)
    phi_cms_h1_b1 = b_cms_h1_detspace[:, 2]

    # Calculate tau inputs
    tau_cms_h2 = boost_a_cm_of_b(jax.numpy.concatenate([tau1_corrected, h2], axis=1).T, h2_detspace[:, 3])
    tau_cms_h2_detspace = fourmomentum_to_det_coord(tau_cms_h2.T)
    cos_theta_cms_h2_tau_vis1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([tau_cms_h2[:, :3], h2[:, :3]], axis=1).T)
    phi_cms_h2_tau_vis1 = tau_cms_h2_detspace[:, 2]

    # Calculate dihiggs inputs
    dihiggs_system = four_vec_sum(jax.numpy.concatenate([h1, h2], axis=1).T)
    dihiggs_system_detspace = fourmomentum_to_det_coord(dihiggs_system.T)
    dihiggs_mass = dihiggs_system_detspace[:, 3]
    dihiggs_system_pt = dihiggs_system_detspace[:, 0]
    dihiggs_system_pz = dihiggs_system[:, 2]
    dihiggs_system_phi = dihiggs_system_detspace[:, 2]

    # Calculate h1 inputs
    h1_cms_dihiggs = boost_a_cm_of_b(jax.numpy.concatenate([h1, dihiggs_system], axis=1).T, dihiggs_mass)
    h1_cms_dihiggs_detspace = fourmomentum_to_det_coord(h1_cms_dihiggs.T)
    cos_theta_h1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([h1_cms_dihiggs[:, :3], dihiggs_system[:, :3]], axis=1).T)
    phi_h1 = h1_cms_dihiggs_detspace[:, 2]

    # grad_func = jax.grad(calculate_invariant_mass)
    # batched_grad = jax.vmap(grad_func)
    # batch_inputs = jax.numpy.stack(
    #     [
    #         ak.to_numpy(b_corrected_vec[:, 0].x).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 0].y).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 0].z).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 0].t).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 1].x).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 1].y).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 1].z).filled(),
    #         ak.to_numpy(b_corrected_vec[:, 1].t).filled(),
    #         ak.to_numpy(taus_corr[:, 0].x),
    #         ak.to_numpy(taus_corr[:, 0].y),
    #         ak.to_numpy(taus_corr[:, 0].z),
    #         ak.to_numpy(taus_corr[:, 0].t),
    #         ak.to_numpy(taus_corr[:, 1].x),
    #         ak.to_numpy(taus_corr[:, 1].y),
    #         ak.to_numpy(taus_corr[:, 1].z),
    #         ak.to_numpy(taus_corr[:, 1].t),
    #     ], axis=1
    # )
    # grads = batched_grad(batch_inputs)
    # test_inputs = jax.numpy.stack([
    #     ak.to_numpy(b_corrected_vec[0, 0].px),
    #     ak.to_numpy(b_corrected_vec[0, 0].py),
    #     ak.to_numpy(b_corrected_vec[0, 0].pz),
    #     ak.to_numpy(b_corrected_vec[0, 0].e),
    #     ak.to_numpy(taus_corr[0, 0].px),
    #     ak.to_numpy(taus_corr[0, 0].y),
    #     ak.to_numpy(taus_corr[0, 0].pz),
    #     ak.to_numpy(taus_corr[0, 0].e),
    #     ak.to_numpy(b_corrected_vec[0, 0].mass),
    #     ak.to_numpy(taus_corr[0, 0].mass),
    # ], axis=0)

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
    events = set_ak_column(
        events, "pdf_input_vars_reco_higgs", pdf_input_vars_reco_higgs)

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
    tau_charge_mask = ak.argsort(events.Tau.charge, axis=1, ascending=False)
    taus_sorted = events.Tau[tau_charge_mask]
    taus_sorted = ak.pad_none(taus_sorted, 2, axis=1, clip=True)

    # 6 Systems: Detector, t_vis t_vis, t_vis1, t_vis2, W+, W-
    # t_vis = tau + b

    # b-Assignment
    # Try that M_lb^2 <= M_t^2 - M_W^2, assign b according to that
    m_tw_sq = 175**2 - 80.3**2
    rng = np.random.default_rng()
    which_b = rng.integers(0, 1, endpoint=True, size=len(events))
    b1_random = ak.where(
        which_b == 0, events.HHBJet[:, 0], events.HHBJet[:, 1])
    b2_random = ak.where(
        which_b == 0, events.HHBJet[:, 1], events.HHBJet[:, 0])
    # Assignment one:
    t11_vis = taus_sorted[:, 0].add(b1_random)
    t22_vis = taus_sorted[:, 1].add(b2_random)
    M_11_sq = t11_vis.mass**2
    M_22_sq = t22_vis.mass**2
    hyp1_mask = ak.concatenate([            # True if constraint not fulfilled
        ak.Array(ak.fill_none(M_11_sq, False) > m_tw_sq)[:, None],
        ak.Array(ak.fill_none(M_22_sq, False) > m_tw_sq)[:, None],
    ], axis=1)
    # True if at least one t_vis doesn't fulfill constraint
    hyp1_mask = ak.any(hyp1_mask, axis=1)
    # Assingment two:
    t12_vis = taus_sorted[:, 0].add(b2_random)
    t21_vis = taus_sorted[:, 1].add(b1_random)
    M_12_sq = t12_vis.mass**2
    M_21_sq = t21_vis.mass**2
    hyp2_mask = ak.concatenate([
        ak.Array(ak.fill_none(M_12_sq, False) > m_tw_sq)[:, None],
        ak.Array(ak.fill_none(M_21_sq, False) > m_tw_sq)[:, None],
    ], axis=1)
    hyp2_mask = ak.any(hyp2_mask, axis=1)
    # If the constraint is not fulfilled in one of the two assignments, but is fulfilled in the other, take that one
    # If both are false, take the one with less deviation
    # Make sure that deviation is only calculated if M_bl_squared is bigger than M_t^2 - M_W^2
    both_false_mask = ak.all(ak.concatenate(
        [hyp1_mask[:, None], hyp2_mask[:, None]], axis=1), axis=1)
    deviations1 = return_deviations_where_necessary(
        M_11_sq, M_22_sq, both_false_mask)
    deviations2 = return_deviations_where_necessary(
        M_12_sq, M_21_sq, both_false_mask)
    choose_hyp1 = deviations1 < deviations2
    # Take t1_vis from assignment 2 if assignment 1 does not fulfill constraint and if at least one of both does
    # fulfill the constraint
    t1_vis = ak.mask(ak.where(hyp1_mask, t12_vis, t11_vis),
                     both_false_mask == bool(0))
    # only meaningfull if both_false_mask is True
    t1_vis_both_false = ak.where(choose_hyp1, t11_vis, t12_vis)
    t1_vis = ak.where(both_false_mask, t1_vis_both_false, t1_vis)
    # Take t2_vis from assignment 2 if assignment 1 does not fulfill constraint and if at least one of both does
    # fulfill the constraint
    t2_vis = ak.mask(ak.where(hyp1_mask, t21_vis, t22_vis),
                     both_false_mask == bool(0))
    # only meaningfull if both_false_mask is True
    t2_vis_both_false = ak.where(choose_hyp1, t22_vis, t21_vis)
    t2_vis = ak.where(both_false_mask, t2_vis_both_false, t2_vis)
    # b assignment done

    # Detector System
    tt_vis_system = t1_vis.add(t2_vis)
    tt_vis_system_mass = tt_vis_system.absolute()
    tt_vis_system_pt = tt_vis_system.pt
    tt_vis_system_pz = tt_vis_system.pz
    tt_vis_system_phi = tt_vis_system.phi

    # tt_vis system
    t1_vis_cms_tt_vis = t1_vis.boostCM_of(tt_vis_system)
    t2_vis_cms_tt_vis = t2_vis.boostCM_of(tt_vis_system)
    t_vis_y_diff = calculate_rapidity(
        t1_vis_cms_tt_vis) - calculate_rapidity(t2_vis_cms_tt_vis)
    t1_vis_phi = t1_vis_cms_tt_vis.phi

    # t1_vis system (tbar)
    tau1_cms_t1_vis = taus_sorted[:, 0].boostCM_of(t1_vis)
    tau1_cos_theta_star_cms_t1_vis = signed_cos_deltaangle(
        tau1_cms_t1_vis, t1_vis)
    tau1_phi = tau1_cms_t1_vis.phi

    # t2_vis system (tbar)
    tau2_cms_t2_vis = taus_sorted[:, 1].boostCM_of(t2_vis)
    tau2_cos_theta_star_cms_t2_vis = signed_cos_deltaangle(
        tau2_cms_t2_vis, t2_vis)
    tau2_phi = tau2_cms_t2_vis.phi

    # W^+ system
    M_lb1 = t1_vis.absolute()
    tau1_cos_theta_star_cms_wplus = 2 * \
        M_lb1**2 / (175**2 - 80.3**2 - 1.7**2) - 1

    # W^- system
    M_lb2 = t2_vis.absolute()
    tau2_cos_theta_star_cms_wminus = 2 * \
        M_lb2**2 / (175**2 - 80.3**2 - 1.7**2) - 1
    pdf_input_vars_reco_top = ak.zip({
        "tt_vis_system_mass": tt_vis_system_mass,
        "tt_vis_system_pt": tt_vis_system_pt,
        "tt_vis_system_pz": tt_vis_system_pz,
        "tt_vis_system_phi": tt_vis_system_phi,
        "t_vis_y_diff": t_vis_y_diff,
        "t1_vis_phi": t1_vis_phi,
        "tau1_cos_theta_star_cms_t1_vis": tau1_cos_theta_star_cms_t1_vis,
        "tau1_phi": tau1_phi,
        "tau2_cos_theta_star_cms_t2_vis": tau2_cos_theta_star_cms_t2_vis,
        "tau2_phi": tau2_phi,
        "tau1_cos_theta_star_cms_wplus": tau1_cos_theta_star_cms_wplus,
        "tau2_cos_theta_star_cms_wminus": tau2_cos_theta_star_cms_wminus,
    }, with_name="pdf_input_vars_reco_top")
    pdf_input_vars_reco_top = ak.mask(
        pdf_input_vars_reco_top, events.channel_id == 3)
    events = set_ak_column(
        events, "pdf_input_vars_reco_top", pdf_input_vars_reco_top)

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
