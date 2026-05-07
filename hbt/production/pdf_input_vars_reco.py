from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
import jax

np = maybe_import("numpy")
ak = maybe_import("awkward")


def calculate_rapidity(particle):
    particle_y = 1 / 2 * np.log(
        (particle.energy + particle.pz) / (particle.energy - particle.pz))
    particle_y = np.nan_to_num(particle_y)
    return particle_y


def calculate_rapidity_for_jax(inputs):
    """Calculates rapidity for given object
    Args:
        inputs: Array with the four-momentum components of the objects, in order px, py, pz, energy
    Returns:
        rapidity: Rapidity of the object
    """
    pz = inputs[2]
    energy = inputs[3]
    y = 1 / 2 * jax.numpy.log(
        (energy + pz) / (energy - pz))
    particle_y = jax.numpy.nan_to_num(y, nan=-99999.0)
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
    abs_a = jax.numpy.sqrt(ax**2 + ay**2 + az**2)
    abs_b = jax.numpy.sqrt(bx**2 + by**2 + bz**2)
    return a_times_b / (abs_a * abs_b)


def calculate_constraint_term(
    corrected_distr1,
        corrected_distr2,
        uncorr_distr1,
        uncorr_distr2,
        mean_sigma_correction_array,
):
    """Calculates log of Likelihood that indicates the probability of a measurement agreeing with the correction
    """
    mean_correction1, mean_correction2 = mean_sigma_correction_array[0], mean_sigma_correction_array[1]
    sigma_correction1, sigma_correction2 = mean_sigma_correction_array[2], mean_sigma_correction_array[3]

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
    ], axis=0)


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
    ], axis=0)


def four_vec_sum(inputs):
    px1, py1, pz1, e1 = inputs[0], inputs[1], inputs[2], inputs[3]
    px2, py2, pz2, e2 = inputs[4], inputs[5], inputs[6], inputs[7]
    px = px1 + px2
    py = py1 + py2
    pz = pz1 + pz2
    e = e1 + e2
    return jax.numpy.stack([
        px, py, pz, e,
    ], axis=0)


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
        p_vec_a + (gamma_b - 1) * (jax.numpy.linalg.vecdot(p_vec_a, normal_n, axis=0)) *
        normal_n - gamma_b * norm_v * a_energy * normal_n
    )
    return jax.numpy.stack([
        boosted_p[0], boosted_p[1], boosted_p[2], boosted_e,
    ], axis=0)


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


# ------------------ Helper functions to get the relevant particles ----------------------------
def calculate_b_corrected(inputs, mean_sigma_correction_array, calculate_constr_term=False):
    b1_pt, b1_eta, b1_phi, b1_mass = inputs[0], inputs[1], inputs[2], inputs[3]
    b2_pt, b2_eta, b2_phi, b2_mass = inputs[4], inputs[5], inputs[6], inputs[7]
    b1_inputs = jax.numpy.stack([
        b1_pt, b1_eta, b1_phi, b1_mass,
    ], axis=0)
    b2_inputs = jax.numpy.stack([
        b2_pt, b2_eta, b2_phi, b2_mass,
    ], axis=0)

    b1 = det_coords_to_fourmomentum(b1_inputs)
    b2 = det_coords_to_fourmomentum(b2_inputs)

    m_bb_rec = calculate_invariant_mass(
        jax.numpy.concatenate([b1, b2], axis=0))

    # Calculate correction terms: b
    m_bb = 125
    b1_corrected = jax.numpy.stack([
        b1[0] * m_bb / m_bb_rec,
        b1[1] * m_bb / m_bb_rec,
        b1[2] * m_bb / m_bb_rec,
        b1[3] * m_bb / m_bb_rec,

    ], axis=0)
    b2_corrected = jax.numpy.stack([
        b2[0] * m_bb / m_bb_rec,
        b2[1] * m_bb / m_bb_rec,
        b2[2] * m_bb / m_bb_rec,
        b2[3] * m_bb / m_bb_rec,

    ], axis=0)
    if not calculate_constr_term:
        b1_corrected = fourmomentum_to_det_coord(b1_corrected)
        b2_corrected = fourmomentum_to_det_coord(b2_corrected)
        return jax.numpy.concatenate([b1_corrected, b2_corrected], axis=0)
    else:
        constr_term_b = calculate_constraint_term(
            fourmomentum_to_det_coord(b1_corrected)[0],
            fourmomentum_to_det_coord(b2_corrected)[0],
            b1_pt,
            b2_pt,
            mean_sigma_correction_array,
        )
        return constr_term_b


def calculate_tau_corrected(inputs, mean_sigma_correction_array, calculate_constr_term=False):
    tau1_pt, tau1_eta, tau1_phi, tau1_mass = inputs[8], inputs[9], inputs[10], inputs[11]
    tau2_pt, tau2_eta, tau2_phi, tau2_mass = inputs[12], inputs[13], inputs[14], inputs[15]
    tau1_inputs = jax.numpy.stack([
        tau1_pt, tau1_eta, tau1_phi, tau1_mass,
    ], axis=0)
    tau2_inputs = jax.numpy.stack([
        tau2_pt, tau2_eta, tau2_phi, tau2_mass,
    ], axis=0)
    tau1 = det_coords_to_fourmomentum(tau1_inputs)
    tau2 = det_coords_to_fourmomentum(tau2_inputs)

    m_tautau = 125
    m_tautau_rec = calculate_invariant_mass(
        jax.numpy.concatenate([tau1, tau2], axis=0))
    m_tautau_rec = jax.numpy.nan_to_num(m_tautau_rec, nan=-99999.)
    tau1_corrected = jax.numpy.stack([
        tau1[0] * m_tautau / m_tautau_rec,
        tau1[1] * m_tautau / m_tautau_rec,
        tau1[2] * m_tautau / m_tautau_rec,
        tau1[3] * m_tautau / m_tautau_rec,

    ], axis=0)
    tau2_corrected = jax.numpy.stack([
        tau2[0] * m_tautau / m_tautau_rec,
        tau2[1] * m_tautau / m_tautau_rec,
        tau2[2] * m_tautau / m_tautau_rec,
        tau2[3] * m_tautau / m_tautau_rec,

    ], axis=0)
    if not calculate_constr_term:
        tau1_corrected = fourmomentum_to_det_coord(tau1_corrected)
        tau2_corrected = fourmomentum_to_det_coord(tau2_corrected)
        return jax.numpy.concatenate([tau1_corrected, tau2_corrected], axis=0)
    else:
        constr_term_tau = calculate_constraint_term(
            fourmomentum_to_det_coord(tau1_corrected)[0],
            fourmomentum_to_det_coord(tau2_corrected)[0],
            tau1_pt,
            tau2_pt,
            mean_sigma_correction_array,
        )
        return constr_term_tau


def calculate_hbb(inputs):
    b1 = det_coords_to_fourmomentum(inputs[:4])
    b2 = det_coords_to_fourmomentum(inputs[4:8])
    b1b2 = jax.numpy.concatenate([b1, b2], axis=0)
    h1 = four_vec_sum(b1b2)
    return h1


def calculate_htautau(inputs):
    tau1 = det_coords_to_fourmomentum(inputs[8:12])
    tau2 = det_coords_to_fourmomentum(inputs[12:])
    tau1tau2 = jax.numpy.concatenate([tau1, tau2], axis=0)
    h2 = four_vec_sum(tau1tau2)
    return h2


def calculate_dihiggs_system(inputs):
    h1 = calculate_hbb(inputs)
    h2 = calculate_htautau(inputs)
    dihiggs_system = four_vec_sum(jax.numpy.concatenate([h1, h2], axis=0))
    return dihiggs_system


# Helper functions for ttbar likelihood
def calculate_t_vis(inputs):
    """Calculate t_vis / lb for both b assignment possibilities, choose the one with M_lb^2 <= M_t^2 - M_W^2
    if possible, else choose the one where the deviation is smaller
    Args:
        inputs: Array with bbtautau pt, eta, phi, m
    Returns:
        tvis: Array with t1_vis, t2_vis px, py, pz, energy
    """
    m_tw_sq = 175**2 - 80.3**2

    b1 = det_coords_to_fourmomentum(inputs[:4])
    b2 = det_coords_to_fourmomentum(inputs[4:8])
    tau1 = det_coords_to_fourmomentum(inputs[8:12])
    tau2 = det_coords_to_fourmomentum(inputs[12:])

    # Hypothesis 1
    t1_vis1 = four_vec_sum(jax.numpy.concatenate([b1, tau1], axis=0))
    t2_vis1 = four_vec_sum(jax.numpy.concatenate([b2, tau2], axis=0))
    M_11_sq = calculate_invariant_mass(jax.numpy.concatenate([b1, tau1], axis=0))**2
    M_21_sq = calculate_invariant_mass(jax.numpy.concatenate([b2, tau2], axis=0))**2

    # Hypothesis 2
    t1_vis2 = four_vec_sum(jax.numpy.concatenate([b2, tau1], axis=0))
    t2_vis2 = four_vec_sum(jax.numpy.concatenate([b1, tau2], axis=0))
    M_12_sq = calculate_invariant_mass(jax.numpy.concatenate([b2, tau1], axis=0))**2
    M_22_sq = calculate_invariant_mass(jax.numpy.concatenate([b1, tau2], axis=0))**2

    # hyp1_mask is True if hyp1 violates constraints, same for hyp2
    hyp1_mask = jax.numpy.any(
        jax.numpy.array([M_11_sq, M_21_sq]) > m_tw_sq,
        axis=0,
    )

    # Create operand dict
    operands = {
        "M_11_sq": M_11_sq,
        "M_21_sq": M_21_sq,
        "M_12_sq": M_12_sq,
        "M_22_sq": M_22_sq,
        "t1_vis1": t1_vis1,
        "t2_vis1": t2_vis1,
        "t1_vis2": t1_vis2,
        "t2_vis2": t2_vis2,
    }
    hyp1_mask = jax.numpy.any(
        jax.numpy.array([M_11_sq, M_21_sq]) > m_tw_sq,
        axis=0,
    )

    def take_hyp2(operands):
        return operands["t1_vis2"], operands["t2_vis2"]

    def take_hyp1(operands):
        return operands["t1_vis1"], operands["t2_vis1"]

    def true_fun_level2(operands):
        M_12_sq = operands["M_12_sq"]
        M_22_sq = operands["M_22_sq"]
        M_21_sq = operands["M_21_sq"]
        M_11_sq = operands["M_11_sq"]
        diff_arr1 = jax.numpy.array([M_11_sq, M_21_sq]) - m_tw_sq
        diff_arr1 = jax.numpy.nan_to_num(diff_arr1, nan=-99999.9)
        # only look at deviations that violate the constraint, take hypothesis with smaller deviation
        pos_deviations1 = jax.numpy.where(diff_arr1 > 0, diff_arr1, jax.numpy.zeros_like(diff_arr1))
        mean_diff1 = jax.numpy.mean(pos_deviations1)
        diff_arr2 = jax.numpy.array([M_12_sq, M_22_sq]) - m_tw_sq
        diff_arr2 = jax.numpy.nan_to_num(diff_arr2, nan=-99999.9)
        pos_deviations2 = jax.numpy.where(diff_arr2 > 0, diff_arr2, jax.numpy.zeros_like(diff_arr2))
        mean_diff2 = jax.numpy.mean(pos_deviations2)
        # True -> take hyp 1
        mean_diff_mask = mean_diff1 < mean_diff2
        t1_vis, t2_vis = jax.lax.cond(mean_diff_mask, take_hyp1, take_hyp2, operands)
        return t1_vis, t2_vis

    def true_fun_level1(operands):
        M_12_sq = operands["M_12_sq"]
        M_22_sq = operands["M_22_sq"]
        m_tw_sq = 175**2 - 80.3**2
        hyp2_mask = jax.numpy.any(
            jax.numpy.array([M_12_sq, M_22_sq]) > m_tw_sq,
            axis=0,
        )

        t1_vis, t2_vis = jax.lax.cond(hyp2_mask, true_fun_level2, take_hyp2, operands)
        return t1_vis, t2_vis

    t1_vis, t2_vis = jax.lax.cond(hyp1_mask, true_fun_level1, take_hyp1, operands)

    # if hyp1_mask:
    #     hyp2_mask = jax.numpy.any(
    #         jax.numpy.array([M_12_sq, M_22_sq]) > m_tw_sq,
    #         axis=0,
    #     )
    #     if hyp2_mask:
    #         # choose hypothesis with less (positive) deviation from m_tw_sq
    #         diff_arr1 = jax.numpy.array([M_11_sq, M_21_sq]) - m_tw_sq
    #         mean_diff1 = jax.numpy.mean(diff_arr1[diff_arr1 > 0])
    #         diff_arr2 = jax.numpy.array([M_12_sq, M_22_sq]) - m_tw_sq
    #         mean_diff2 = jax.numpy.mean(diff_arr2[diff_arr2 > 0])
    #         if mean_diff1 < mean_diff2:
    #             t1_vis, t2_vis = t1_vis1, t2_vis1
    #         else:
    #             t1_vis, t2_vis = t1_vis2, t2_vis2
    #     else:
    #         t1_vis, t2_vis = t1_vis2, t2_vis2
    # else:
    #     t1_vis, t2_vis = t1_vis1, t2_vis1

    return jax.numpy.concatenate([t1_vis, t2_vis], axis=0)


def calculate_tt_vis_system(inputs):
    tvis = calculate_t_vis(inputs)
    tt_vis_system = four_vec_sum(tvis)
    return tt_vis_system


# ----------------------------- Scalar output functions to calculate the likelihood inputs ------------------------
# Inputs for Higgs likelihood
def calculate_cos_theta_cms_h1_b1(inputs):
    b1_corrected = det_coords_to_fourmomentum(inputs[:4])
    h1 = calculate_hbb(inputs)
    h1_detspace = fourmomentum_to_det_coord(h1)
    b_cms_h1 = boost_a_cm_of_b(jax.numpy.concatenate([b1_corrected, h1], axis=0), h1_detspace[3])
    cos_theta_cms_h1_b1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([b_cms_h1[:3], h1[:3]], axis=0))
    return cos_theta_cms_h1_b1


def calculate_phi_cms_h1_b1(inputs):
    b1_corrected = det_coords_to_fourmomentum(inputs[:4])
    h1 = calculate_hbb(inputs)
    h1_detspace = fourmomentum_to_det_coord(h1)
    b_cms_h1 = boost_a_cm_of_b(jax.numpy.concatenate([b1_corrected, h1], axis=0), h1_detspace[3])
    b_cms_h1_detspace = fourmomentum_to_det_coord(b_cms_h1)
    phi_cms_h1_b1 = b_cms_h1_detspace[2]
    return phi_cms_h1_b1


def calculate_cos_theta_cms_h2_tau_vis1(inputs):
    tau1_corrected = det_coords_to_fourmomentum(inputs[8:12])
    h2 = calculate_htautau(inputs)
    h2_detspace = fourmomentum_to_det_coord(h2)
    tau_cms_h2 = boost_a_cm_of_b(jax.numpy.concatenate([tau1_corrected, h2], axis=0), h2_detspace[3])
    cos_theta_cms_h2_tau_vis1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([tau_cms_h2[:3], h2[:3]], axis=0))
    return cos_theta_cms_h2_tau_vis1


def calculate_phi_cms_h2_tau_vis1(inputs):
    tau1_corrected = det_coords_to_fourmomentum(inputs[8:12])
    h2 = calculate_htautau(inputs)
    h2_detspace = fourmomentum_to_det_coord(h2)
    tau_cms_h2 = boost_a_cm_of_b(jax.numpy.concatenate([tau1_corrected, h2], axis=0), h2_detspace[3])
    tau_cms_h2_detspace = fourmomentum_to_det_coord(tau_cms_h2)
    phi_cms_h2_tau_vis1 = tau_cms_h2_detspace[2]
    return phi_cms_h2_tau_vis1


def calculate_dihiggs_mass(inputs):
    dihiggs_system = calculate_dihiggs_system(inputs)
    dihiggs_system_detspace = fourmomentum_to_det_coord(dihiggs_system)
    return dihiggs_system_detspace[3]


def calculate_dihiggs_system_pt(inputs):
    dihiggs_system = calculate_dihiggs_system(inputs)
    dihiggs_system_detspace = fourmomentum_to_det_coord(dihiggs_system)
    return dihiggs_system_detspace[0]


def calculate_dihiggs_system_pz(inputs):
    dihiggs_system = calculate_dihiggs_system(inputs)
    return dihiggs_system[2]


def calculate_dihiggs_system_phi(inputs):
    dihiggs_system = calculate_dihiggs_system(inputs)
    dihiggs_system_detspace = fourmomentum_to_det_coord(dihiggs_system)
    return dihiggs_system_detspace[2]


def calculate_cos_theta_h1(inputs):
    h1 = calculate_hbb(inputs)
    dihiggs_system = calculate_dihiggs_system(inputs)
    h1_cms_dihiggs = boost_a_cm_of_b(jax.numpy.concatenate([h1, dihiggs_system], axis=0), dihiggs_system[3])
    cos_theta_h1 = signed_cos_deltaangle_for_jax(jax.numpy.concatenate([h1_cms_dihiggs[:3], dihiggs_system[:3]], axis=0))
    return cos_theta_h1


def calculate_phi_h1(inputs):
    h1 = calculate_hbb(inputs)
    dihiggs_system = calculate_dihiggs_system(inputs)
    h1_cms_dihiggs = boost_a_cm_of_b(jax.numpy.concatenate([h1, dihiggs_system], axis=0), dihiggs_system[3])
    h1_cms_dihiggs_detspace = fourmomentum_to_det_coord(h1_cms_dihiggs)
    return h1_cms_dihiggs_detspace[2]


def calculate_mean_correction(inputs, ch_id_mask):
    """Calculates the mean correction and std dev of correction for all relevant events (e.g., ch_id_mask applied)
    Can be done outside of gradient collection / batching per event, as all events need to be taken into account.
    """
    part1_pt, part1_eta, part1_phi, part1_mass = inputs[0], inputs[1], inputs[2], inputs[3]
    part2_pt, part2_eta, part2_phi, part2_mass = inputs[4], inputs[5], inputs[6], inputs[7]
    part1_inputs = jax.numpy.stack([
        part1_pt, part1_eta, part1_phi, part1_mass,
    ], axis=0)
    part2_inputs = jax.numpy.stack([
        part2_pt, part2_eta, part2_phi, part2_mass,
    ], axis=0)

    part1 = det_coords_to_fourmomentum(part1_inputs)
    part2 = det_coords_to_fourmomentum(part2_inputs)

    m_inv_rec = calculate_invariant_mass(
        jax.numpy.concatenate([part1, part2], axis=0))

    # Calculate correction terms: b
    m_inv = 125
    part1_corrected = jax.numpy.stack([
        part1[0] * m_inv / m_inv_rec,
        part1[1] * m_inv / m_inv_rec,
        part1[2] * m_inv / m_inv_rec,
        part1[3] * m_inv / m_inv_rec,

    ], axis=0)
    part2_corrected = jax.numpy.stack([
        part2[0] * m_inv / m_inv_rec,
        part2[1] * m_inv / m_inv_rec,
        part2[2] * m_inv / m_inv_rec,
        part2[3] * m_inv / m_inv_rec,

    ], axis=0)
    part1_corrected = fourmomentum_to_det_coord(part1_corrected)
    part2_corrected = fourmomentum_to_det_coord(part2_corrected)

    mean_correction1 = jax.numpy.mean(part1_corrected[0][ch_id_mask] - part1_pt[ch_id_mask])
    mean_correction2 = jax.numpy.mean(part2_corrected[0][ch_id_mask] - part2_pt[ch_id_mask])
    sigma_correction1 = jax.numpy.std(part1_corrected[0][ch_id_mask] - part1_pt[ch_id_mask])
    sigma_correction2 = jax.numpy.std(part2_corrected[0][ch_id_mask] - part2_pt[ch_id_mask])
    return jax.numpy.array([mean_correction1, mean_correction2, sigma_correction1, sigma_correction2])


def calculate_constr_term_b(inputs, mean_sigma_correction_array):
    return calculate_b_corrected(
        inputs, mean_sigma_correction_array, calculate_constr_term=True,
    )


def calculate_constr_term_tau(inputs, mean_sigma_correction_array):
    return calculate_tau_corrected(
        inputs, mean_sigma_correction_array, calculate_constr_term=True,
    )


# Inputs for ttbar-likelihood
def calculate_tt_vis_system_mass(inputs):
    tt_vis_system = calculate_tt_vis_system(inputs)
    tt_vis_system_mass = fourmomentum_to_det_coord(tt_vis_system)[3]
    return tt_vis_system_mass


def calculate_tt_vis_system_pt(inputs):
    tt_vis_system = calculate_tt_vis_system(inputs)
    tt_vis_system_pt = fourmomentum_to_det_coord(tt_vis_system)[0]
    return tt_vis_system_pt


def calculate_tt_vis_system_pz(inputs):
    tt_vis_system = calculate_tt_vis_system(inputs)
    return tt_vis_system[2]


def calculate_tt_vis_system_phi(inputs):
    tt_vis_system = calculate_tt_vis_system(inputs)
    tt_vis_system_phi = fourmomentum_to_det_coord(tt_vis_system)[2]
    return tt_vis_system_phi


def calculate_t_vis_y_diff(inputs):
    tvis = calculate_t_vis(inputs)
    tt_vis_system = calculate_tt_vis_system(inputs)
    tt_vis_system_mass = fourmomentum_to_det_coord(tt_vis_system)[3]
    t1_vis = tvis[:4]
    t2_vis = tvis[4:]
    t1_vis_cms_tt_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([t1_vis, tt_vis_system]),
        tt_vis_system_mass,
    )
    t2_vis_cms_tt_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([t2_vis, tt_vis_system]),
        tt_vis_system_mass,
    )
    t_vis_y_diff = calculate_rapidity_for_jax(
        t1_vis_cms_tt_vis) - calculate_rapidity_for_jax(t2_vis_cms_tt_vis)
    return t_vis_y_diff


def calculate_t1_vis_phi(inputs):
    tvis = calculate_t_vis(inputs)
    tt_vis_system = calculate_tt_vis_system(inputs)
    tt_vis_system_mass = fourmomentum_to_det_coord(tt_vis_system)[3]
    t1_vis = tvis[:4]
    t1_vis_cms_tt_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([t1_vis, tt_vis_system]),
        tt_vis_system_mass,
    )
    t1_vis_phi = fourmomentum_to_det_coord(t1_vis_cms_tt_vis)[2]
    return t1_vis_phi


def calculate_tau1_cos_theta_star_cms_t1_vis(inputs):
    tau1 = det_coords_to_fourmomentum(inputs[8:12])
    t1_vis = calculate_t_vis(inputs)[:4]
    t1_vis_mass = fourmomentum_to_det_coord(t1_vis)[3]
    tau1_cms_t1_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([tau1, t1_vis]),
        t1_vis_mass,
    )
    tau1_cos_theta_star_cms_t1_vis = signed_cos_deltaangle_for_jax(
        jax.numpy.concatenate([tau1_cms_t1_vis[:3], t1_vis[:3]], axis=0),
    )
    return tau1_cos_theta_star_cms_t1_vis


def calculate_tau2_cos_theta_star_cms_t2_vis(inputs):
    tau2 = det_coords_to_fourmomentum(inputs[12:])
    t2_vis = calculate_t_vis(inputs)[4:]
    t2_vis_mass = fourmomentum_to_det_coord(t2_vis)[3]
    tau2_cms_t2_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([tau2, t2_vis]),
        t2_vis_mass,
    )
    tau2_cos_theta_star_cms_t2_vis = signed_cos_deltaangle_for_jax(
        jax.numpy.concatenate([tau2_cms_t2_vis[:3], t2_vis[:3]], axis=0),
    )
    return tau2_cos_theta_star_cms_t2_vis


def calculate_tau1_phi(inputs):
    tau1 = det_coords_to_fourmomentum(inputs[8:12])
    t1_vis = calculate_t_vis(inputs)[:4]
    t1_vis_mass = fourmomentum_to_det_coord(t1_vis)[3]
    tau1_cms_t1_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([tau1, t1_vis]),
        t1_vis_mass,
    )
    tau1_phi = fourmomentum_to_det_coord(tau1_cms_t1_vis)[2]
    return tau1_phi


def calculate_tau2_phi(inputs):
    tau2 = det_coords_to_fourmomentum(inputs[12:])
    t2_vis = calculate_t_vis(inputs)[4:]
    t2_vis_mass = fourmomentum_to_det_coord(t2_vis)[3]
    tau2_cms_t2_vis = boost_a_cm_of_b(
        jax.numpy.concatenate([tau2, t2_vis]),
        t2_vis_mass,
    )
    tau2_phi = fourmomentum_to_det_coord(tau2_cms_t2_vis)[2]
    return tau2_phi


def calculate_tau1_cos_theta_star_cms_wplus(inputs):
    t1_vis = calculate_t_vis(inputs)[:4]
    m_lb1 = fourmomentum_to_det_coord(t1_vis)[3]
    tau1_cos_theta_star_cms_wplus = 2 * \
        m_lb1**2 / (175**2 - 80.3**2 - 1.7**2) - 1
    return tau1_cos_theta_star_cms_wplus


def calculate_tau2_cos_theta_star_cms_wminus(inputs):
    t2_vis = calculate_t_vis(inputs)[4:]
    m_lb2 = fourmomentum_to_det_coord(t2_vis)[3]
    tau2_cos_theta_star_cms_wminus = 2 * \
        m_lb2**2 / (175**2 - 80.3**2 - 1.7**2) - 1
    return tau2_cos_theta_star_cms_wminus


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
    ], axis=0, dtype=jax.numpy.float64)
    b2_inputs = jax.numpy.stack([
        b_pt[:, 1], b_eta[:, 1], b_phi[:, 1], b_mass[:, 1],
    ], axis=0, dtype=jax.numpy.float64)
    tau1_inputs = jax.numpy.stack([
        tau_pt[:, 0], tau_eta[:, 0], tau_phi[:, 0], tau_mass[:, 0],
    ], axis=0, dtype=jax.numpy.float64)
    tau2_inputs = jax.numpy.stack([
        tau_pt[:, 1], tau_eta[:, 1], tau_phi[:, 1], tau_mass[:, 1],
    ], axis=0, dtype=jax.numpy.float64)

    inputs = jax.numpy.concatenate([
        b1_inputs, b2_inputs, tau1_inputs, tau2_inputs,
    ], axis=0)

    # Apply constraints:
    mean_sigma_correction_array_b = calculate_mean_correction(inputs[:8], ch_id_mask)
    mean_sigma_correction_array_tau = calculate_mean_correction(inputs[8:], ch_id_mask)
    batched_calculate_b_corrected = jax.vmap(calculate_b_corrected, in_axes=(0, None))
    batched_calculate_tau_corrected = jax.vmap(calculate_tau_corrected, in_axes=(0, None))
    b1_corrected = batched_calculate_b_corrected(inputs.T, mean_sigma_correction_array_b)[:, :4]
    b2_corrected = batched_calculate_b_corrected(inputs.T, mean_sigma_correction_array_b)[:, 4:]
    tau1_corrected = batched_calculate_tau_corrected(inputs.T, mean_sigma_correction_array_tau)[:, :4]
    tau2_corrected = batched_calculate_tau_corrected(inputs.T, mean_sigma_correction_array_tau)[:, 4:]
    # Calculate constraint terms
    batched_calculate_constr_term_b = jax.vmap(calculate_constr_term_b, in_axes=(0, None))
    batched_calculate_constr_term_tau = jax.vmap(calculate_constr_term_tau, in_axes=(0, None))
    constr_term_b = batched_calculate_constr_term_b(inputs.T, mean_sigma_correction_array_b)
    constr_term_tau = batched_calculate_constr_term_tau(inputs.T, mean_sigma_correction_array_tau)
    # Set masses to fixed values: m_b = 4.183 GeV, m_tau = 1.777 Gev
    b1_corrected = b1_corrected.at[:, 3].set(4.183)
    b2_corrected = b2_corrected.at[:, 3].set(4.183)
    tau1_corrected = tau1_corrected.at[:, 3].set(1.777)
    tau2_corrected = tau2_corrected.at[:, 3].set(1.777)
    inputs = jax.numpy.concatenate([
        b1_corrected, b2_corrected, tau1_corrected, tau2_corrected,
    ], axis=1).T

    # ------------------------------- Batching ----------------------------------------------------
    # Calculate inputs with helper functions from above, using jax.vmap, then calculate jacobians
    batched_calculate_dihiggs_mass = jax.vmap(calculate_dihiggs_mass)
    batched_calculate_dihiggs_system_pt = jax.vmap(calculate_dihiggs_system_pt)
    batched_calculate_dihiggs_system_pz = jax.vmap(calculate_dihiggs_system_pz)
    batched_calculate_dihiggs_system_phi = jax.vmap(calculate_dihiggs_system_phi)
    batched_calculate_cos_theta_h1 = jax.vmap(calculate_cos_theta_h1)
    batched_calculate_phi_h1 = jax.vmap(calculate_phi_h1)
    batched_calculate_cos_theta_cms_h2_tau_vis1 = jax.vmap(calculate_cos_theta_cms_h2_tau_vis1)
    batched_calculate_phi_cms_h2_tau_vis1 = jax.vmap(calculate_phi_cms_h2_tau_vis1)
    batched_calculate_cos_theta_cms_h1_b1 = jax.vmap(calculate_cos_theta_cms_h1_b1)
    batched_calculate_phi_cms_h1_b1 = jax.vmap(calculate_phi_cms_h1_b1)

    # Gradient functions
    batched_grad_calculate_dihiggs_mass = jax.vmap(jax.grad(calculate_dihiggs_mass))
    batched_grad_calculate_dihiggs_system_pt = jax.vmap(jax.grad(calculate_dihiggs_system_pt))
    batched_grad_calculate_dihiggs_system_pz = jax.vmap(jax.grad(calculate_dihiggs_system_pz))
    batched_grad_calculate_dihiggs_system_phi = jax.vmap(jax.grad(calculate_dihiggs_system_phi))
    batched_grad_calculate_cos_theta_h1 = jax.vmap(jax.grad(calculate_cos_theta_h1))
    batched_grad_calculate_phi_h1 = jax.vmap(jax.grad(calculate_phi_h1))
    batched_grad_calculate_cos_theta_cms_h2_tau_vis1 = jax.vmap(jax.grad(calculate_cos_theta_cms_h2_tau_vis1))
    batched_grad_calculate_phi_cms_h2_tau_vis1 = jax.vmap(jax.grad(calculate_phi_cms_h2_tau_vis1))
    batched_grad_calculate_cos_theta_cms_h1_b1 = jax.vmap(jax.grad(calculate_cos_theta_cms_h1_b1))
    batched_grad_calculate_phi_cms_h1_b1 = jax.vmap(jax.grad(calculate_phi_cms_h1_b1))
    # batched_grad_calculate_constr_term_b = jax.vmap(jax.grad(calculate_constr_term_b, argnums=0), in_axes=(0, None))
    # batched_grad_calculate_constr_term_tau = jax.vmap(jax.grad(calculate_constr_term_tau, argnums=0),in_axes=(0, None))

    # Calculate likelihood inputs
    dihiggs_mass = batched_calculate_dihiggs_mass(inputs.T)
    dihiggs_system_pt = batched_calculate_dihiggs_system_pt(inputs.T)
    dihiggs_system_pz = batched_calculate_dihiggs_system_pz(inputs.T)
    dihiggs_system_phi = batched_calculate_dihiggs_system_phi(inputs.T)
    cos_theta_h1 = batched_calculate_cos_theta_h1(inputs.T)
    phi_h1 = batched_calculate_phi_h1(inputs.T)
    cos_theta_cms_h2_tau_vis1 = batched_calculate_cos_theta_cms_h2_tau_vis1(inputs.T)
    phi_cms_h2_tau_vis1 = batched_calculate_phi_cms_h2_tau_vis1(inputs.T)
    cos_theta_cms_h1_b1 = batched_calculate_cos_theta_cms_h1_b1(inputs.T)
    phi_cms_h1_b1 = batched_calculate_phi_cms_h1_b1(inputs.T)

    # Calculate Jacobians: Gradient calculation
    # grad_... shape: (Batch_size, 10), 10 is dim of scatt. obs. space and constrained reco space with fixed masses
    grad_dihiggs_mass = batched_grad_calculate_dihiggs_mass(inputs.T)
    grad_dihiggs_system_pt = batched_grad_calculate_dihiggs_system_pt(inputs.T)
    grad_dihiggs_system_pz = batched_grad_calculate_dihiggs_system_pz(inputs.T)
    grad_dihiggs_system_phi = batched_grad_calculate_dihiggs_system_phi(inputs.T)
    grad_cos_theta_h1 = batched_grad_calculate_cos_theta_h1(inputs.T)
    grad_phi_h1 = batched_grad_calculate_phi_h1(inputs.T)
    grad_cos_theta_cms_h2_tau_vis1 = batched_grad_calculate_cos_theta_cms_h2_tau_vis1(inputs.T)
    grad_phi_cms_h2_tau_vis1 = batched_grad_calculate_phi_cms_h2_tau_vis1(inputs.T)
    grad_cos_theta_cms_h1_b1 = batched_grad_calculate_cos_theta_cms_h1_b1(inputs.T)
    grad_phi_cms_h1_b1 = batched_grad_calculate_phi_cms_h1_b1(inputs.T)
    # grad_constr_term_b = batched_grad_calculate_constr_term_b(inputs.T, mean_sigma_correction_array_b)
    # grad_constr_term_tau = batched_grad_calculate_constr_term_tau(inputs.T, mean_sigma_correction_array_tau)

    # Calculate Jacobians: Building the matrices
    # Matrix shape: (Batch_size, num_likelihood_inputs, 10) after removal of unneccesary columns
    jac_matrix = np.concatenate([
        grad_dihiggs_mass[:, None],
        grad_dihiggs_system_pt[:, None],
        grad_dihiggs_system_pz[:, None],
        grad_dihiggs_system_phi[:, None],
        grad_cos_theta_h1[:, None],
        grad_phi_h1[:, None],
        grad_cos_theta_cms_h2_tau_vis1[:, None],
        grad_phi_cms_h2_tau_vis1[:, None],
        grad_cos_theta_cms_h1_b1[:, None],
        grad_phi_cms_h1_b1[:, None],
    ], axis=1, dtype=np.float64)
    # remove mass columns and pt b2, tau2 columns:
    # b1_pt, b1_eta, b1_phi, b1_m, b2_pt, b2_eta, b2_phi, b2_m, tau1_pt, tau1_eta, tau1_phi, tau1_m, tau2_pt, ...
    jac_matrix_constrained_space = jac_matrix[:, :, np.array([0, 1, 2, 5, 6, 8, 9, 10, 13, 14])]
    # squared_matrix = np.matmul(jac_matrix_transposed, jac_matrix)
    # squared_matrix = np.matmul(jac_matrix, jac_matrix_transposed)
    jac_det = np.linalg.det(jac_matrix_constrained_space)
    jac_matrix_transposed = np.transpose(jac_matrix, axes=(0, 2, 1))
    squared_matrix = np.matmul(jac_matrix_transposed, jac_matrix)
    squared_matrix = np.matmul(jac_matrix, jac_matrix_transposed)
    gramsche = np.linalg.det(squared_matrix)
    EMPTY_FLOAT = -99999.9

    # Create the column
    pdf_input_vars_reco_higgs = ak.zip(
        {
            "dihiggs_mass": ak.from_numpy(np.nan_to_num(dihiggs_mass, nan=EMPTY_FLOAT)),
            "dihiggs_system_pt": ak.from_numpy(np.nan_to_num(dihiggs_system_pt, nan=EMPTY_FLOAT)),
            "dihiggs_system_pz": ak.from_numpy(np.nan_to_num(dihiggs_system_pz, nan=EMPTY_FLOAT)),
            "dihiggs_system_phi": ak.from_numpy(np.nan_to_num(dihiggs_system_phi, nan=EMPTY_FLOAT)),
            "cos_theta_h1": ak.from_numpy(np.nan_to_num(cos_theta_h1, nan=EMPTY_FLOAT)),
            "phi_h1": ak.from_numpy(np.nan_to_num(phi_h1, nan=EMPTY_FLOAT)),
            "cos_theta_cms_h2_tau_vis1": ak.from_numpy(np.nan_to_num(cos_theta_cms_h2_tau_vis1, nan=EMPTY_FLOAT)),
            "phi_cms_h2_tau_vis1": ak.from_numpy(np.nan_to_num(phi_cms_h2_tau_vis1, nan=EMPTY_FLOAT)),
            "cos_theta_cms_h1_b1": ak.from_numpy(np.nan_to_num(cos_theta_cms_h1_b1, nan=EMPTY_FLOAT)),
            "phi_cms_h1_b1": ak.from_numpy(np.nan_to_num(phi_cms_h1_b1, nan=EMPTY_FLOAT)),
            "constr_term_tau": ak.from_numpy(np.nan_to_num(constr_term_tau, nan=EMPTY_FLOAT)),
            "constr_term_b": ak.from_numpy(np.nan_to_num(constr_term_b, nan=EMPTY_FLOAT)),
            "jac_det": ak.from_numpy(np.nan_to_num(jac_det, nan=EMPTY_FLOAT)),
            "gramsche": ak.from_numpy(np.nan_to_num(gramsche, nan=EMPTY_FLOAT)),
        },
        with_name="pdf_input_vars_reco_higgs")
    pdf_input_vars_reco_higgs = ak.mask(pdf_input_vars_reco_higgs, ak.from_numpy(np.array(ch_id_mask)))
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

    ch_id_mask = events.channel_id == 3
    ch_id_mask = ak.to_numpy(ch_id_mask)

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
    # b_mass = ak.to_numpy(sorted_bs.mass, allow_missing=False)
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
    # tau_mass = ak.to_numpy(
    #     ak.fill_none(ak.pad_none(sorted_taus.mass, 2, axis=1, clip=True), -99999.0), allow_missing=False,
    # )

    # set masses to fixed values as in higgs function above
    b_mass = np.ones_like(b_phi) * 4.183
    tau_mass = np.ones_like(tau_phi) * 1.777

    # Inputs for higgs input function
    b1_inputs = jax.numpy.stack([
        b_pt[:, 0], b_eta[:, 0], b_phi[:, 0], b_mass[:, 0],
    ], axis=0, dtype=jax.numpy.float64)
    b2_inputs = jax.numpy.stack([
        b_pt[:, 1], b_eta[:, 1], b_phi[:, 1], b_mass[:, 1],
    ], axis=0, dtype=jax.numpy.float64)
    tau1_inputs = jax.numpy.stack([
        tau_pt[:, 0], tau_eta[:, 0], tau_phi[:, 0], tau_mass[:, 0],
    ], axis=0, dtype=jax.numpy.float64)
    tau2_inputs = jax.numpy.stack([
        tau_pt[:, 1], tau_eta[:, 1], tau_phi[:, 1], tau_mass[:, 1],
    ], axis=0, dtype=jax.numpy.float64)

    inputs = jax.numpy.concatenate([
        b1_inputs, b2_inputs, tau1_inputs, tau2_inputs,
    ], axis=0)
    # ------------------------- batching -----------------------------------------
    # Calculate inputs with helper functions from above, using jax.vmap, then calculate jacobians
    batched_calculate_tt_vis_system_mass = jax.vmap(calculate_tt_vis_system_mass)
    batched_calculate_tt_vis_system_pt = jax.vmap(calculate_tt_vis_system_pt)
    batched_calculate_tt_vis_system_pz = jax.vmap(calculate_tt_vis_system_pz)
    batched_calculate_tt_vis_system_phi = jax.vmap(calculate_tt_vis_system_phi)
    batched_calculate_t_vis_y_diff = jax.vmap(calculate_t_vis_y_diff)
    batched_calculate_t1_vis_phi = jax.vmap(calculate_t1_vis_phi)
    batched_calculate_tau1_cos_theta_star_cms_t1_vis = jax.vmap(calculate_tau1_cos_theta_star_cms_t1_vis)
    batched_calculate_tau1_phi = jax.vmap(calculate_tau1_phi)
    batched_calculate_tau2_cos_theta_star_cms_t2_vis = jax.vmap(calculate_tau2_cos_theta_star_cms_t2_vis)
    batched_calculate_tau2_phi = jax.vmap(calculate_tau2_phi)
    batched_calculate_tau1_cos_theta_star_cms_wplus = jax.vmap(calculate_tau1_cos_theta_star_cms_wplus)
    batched_calculate_tau2_cos_theta_star_cms_wminus = jax.vmap(calculate_tau2_cos_theta_star_cms_wminus)

    # Batched gradients
    batched_grad_calculate_tt_vis_system_mass = jax.vmap(jax.grad(calculate_tt_vis_system_mass))
    batched_grad_calculate_tt_vis_system_pt = jax.vmap(jax.grad(calculate_tt_vis_system_pt))
    batched_grad_calculate_tt_vis_system_pz = jax.vmap(jax.grad(calculate_tt_vis_system_pz))
    batched_grad_calculate_tt_vis_system_phi = jax.vmap(jax.grad(calculate_tt_vis_system_phi))
    batched_grad_calculate_t_vis_y_diff = jax.vmap(jax.grad(calculate_t_vis_y_diff))
    batched_grad_calculate_t1_vis_phi = jax.vmap(jax.grad(calculate_t1_vis_phi))
    batched_grad_calculate_tau1_cos_theta_star_cms_t1_vis = jax.vmap(jax.grad(calculate_tau1_cos_theta_star_cms_t1_vis))
    batched_grad_calculate_tau1_phi = jax.vmap(jax.grad(calculate_tau1_phi))
    batched_grad_calculate_tau2_cos_theta_star_cms_t2_vis = jax.vmap(jax.grad(calculate_tau2_cos_theta_star_cms_t2_vis))
    batched_grad_calculate_tau2_phi = jax.vmap(jax.grad(calculate_tau2_phi))
    batched_grad_calculate_tau1_cos_theta_star_cms_wplus = jax.vmap(jax.grad(calculate_tau1_cos_theta_star_cms_wplus))
    batched_grad_calculate_tau2_cos_theta_star_cms_wminus = jax.vmap(jax.grad(calculate_tau2_cos_theta_star_cms_wminus))

    # Calculate likelihood inputs
    tt_vis_system_mass = batched_calculate_tt_vis_system_mass(inputs.T)
    tt_vis_system_pt = batched_calculate_tt_vis_system_pt(inputs.T)
    tt_vis_system_pz = batched_calculate_tt_vis_system_pz(inputs.T)
    tt_vis_system_phi = batched_calculate_tt_vis_system_phi(inputs.T)
    t_vis_y_diff = batched_calculate_t_vis_y_diff(inputs.T)
    t1_vis_phi = batched_calculate_t1_vis_phi(inputs.T)
    tau1_cos_theta_star_cms_t1_vis = batched_calculate_tau1_cos_theta_star_cms_t1_vis(inputs.T)
    tau1_phi = batched_calculate_tau1_phi(inputs.T)
    tau2_cos_theta_star_cms_t2_vis = batched_calculate_tau2_cos_theta_star_cms_t2_vis(inputs.T)
    tau2_phi = batched_calculate_tau2_phi(inputs.T)
    tau1_cos_theta_star_cms_wplus = batched_calculate_tau1_cos_theta_star_cms_wplus(inputs.T)
    tau2_cos_theta_star_cms_wminus = batched_calculate_tau2_cos_theta_star_cms_wminus(inputs.T)

    # Calculate Jacobians: Gradient calculation
    # grad_... shape: (Batch_size, 16), because 16 inputs go into likelihood variable calculation (pt, eta, phi, m) * 4
    grad_tt_vis_system_mass = batched_grad_calculate_tt_vis_system_mass(inputs.T)
    grad_tt_vis_system_pt = batched_grad_calculate_tt_vis_system_pt(inputs.T)
    grad_tt_vis_system_pz = batched_grad_calculate_tt_vis_system_pz(inputs.T)
    grad_tt_vis_system_phi = batched_grad_calculate_tt_vis_system_phi(inputs.T)
    grad_t_vis_y_diff = batched_grad_calculate_t_vis_y_diff(inputs.T)
    grad_t1_vis_phi = batched_grad_calculate_t1_vis_phi(inputs.T)
    grad_tau1_cos_theta_star_cms_t1_vis = batched_grad_calculate_tau1_cos_theta_star_cms_t1_vis(inputs.T)
    grad_tau1_phi = batched_grad_calculate_tau1_phi(inputs.T)
    grad_tau2_cos_theta_star_cms_t2_vis = batched_grad_calculate_tau2_cos_theta_star_cms_t2_vis(inputs.T)
    grad_tau2_phi = batched_grad_calculate_tau2_phi(inputs.T)
    grad_tau1_cos_theta_star_cms_wplus = batched_grad_calculate_tau1_cos_theta_star_cms_wplus(inputs.T)
    grad_tau2_cos_theta_star_cms_wminus = batched_grad_calculate_tau2_cos_theta_star_cms_wminus(inputs.T)

    jac_matrix = np.concatenate([
        grad_tt_vis_system_mass[:, None],
        grad_tt_vis_system_pt[:, None],
        grad_tt_vis_system_pz[:, None],
        grad_tt_vis_system_phi[:, None],
        grad_t_vis_y_diff[:, None],
        grad_t1_vis_phi[:, None],
        grad_tau1_cos_theta_star_cms_t1_vis[:, None],
        grad_tau1_phi[:, None],
        grad_tau2_cos_theta_star_cms_t2_vis[:, None],
        grad_tau2_phi[:, None],
        grad_tau1_cos_theta_star_cms_wplus[:, None],
        grad_tau2_cos_theta_star_cms_wminus[:, None],
    ], axis=1, dtype=np.float64)
    jac_matrix_transposed = np.transpose(jac_matrix, axes=(0, 2, 1))
    jac_matrix_constrained_space = jac_matrix[:, :, np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])]
    squared_matrix = np.matmul(jac_matrix_transposed, jac_matrix)
    squared_matrix = np.matmul(jac_matrix, jac_matrix_transposed)
    jac_det = np.linalg.det(jac_matrix_constrained_space)
    gramsche = np.linalg.det(squared_matrix)

    EMPTY_FLOAT = -99999.9
    pdf_input_vars_reco_top = ak.zip({
        "tt_vis_system_mass": ak.from_numpy(np.nan_to_num(tt_vis_system_mass, nan=EMPTY_FLOAT)),
        "tt_vis_system_pt": ak.from_numpy(np.nan_to_num(tt_vis_system_pt, nan=EMPTY_FLOAT)),
        "tt_vis_system_pz": ak.from_numpy(np.nan_to_num(tt_vis_system_pz, nan=EMPTY_FLOAT)),
        "tt_vis_system_phi": ak.from_numpy(np.nan_to_num(tt_vis_system_phi, nan=EMPTY_FLOAT)),
        "t_vis_y_diff": ak.from_numpy(np.nan_to_num(t_vis_y_diff, nan=EMPTY_FLOAT)),
        "t1_vis_phi": ak.from_numpy(np.nan_to_num(t1_vis_phi, nan=EMPTY_FLOAT)),
        "tau1_cos_theta_star_cms_t1_vis": ak.from_numpy(np.nan_to_num(tau1_cos_theta_star_cms_t1_vis, nan=EMPTY_FLOAT)),
        "tau1_phi": ak.from_numpy(np.nan_to_num(tau1_phi, nan=EMPTY_FLOAT)),
        "tau2_cos_theta_star_cms_t2_vis": ak.from_numpy(np.nan_to_num(tau2_cos_theta_star_cms_t2_vis, nan=EMPTY_FLOAT)),
        "tau2_phi": ak.from_numpy(np.nan_to_num(tau2_phi, nan=EMPTY_FLOAT)),
        "tau1_cos_theta_star_cms_wplus": ak.from_numpy(np.nan_to_num(tau1_cos_theta_star_cms_wplus, nan=EMPTY_FLOAT)),
        "tau2_cos_theta_star_cms_wminus": ak.from_numpy(np.nan_to_num(tau2_cos_theta_star_cms_wminus, nan=EMPTY_FLOAT)),
        "jac_det": ak.from_numpy(np.nan_to_num(jac_det, nan=EMPTY_FLOAT)),
        "gramsche": ak.from_numpy(np.nan_to_num(gramsche, nan=EMPTY_FLOAT)),
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
    events = self[create_pdf_input_vars_reco_top](events, **kwargs)
    events = self[create_pdf_input_vars_reco_higgs](events, **kwargs)
    return events
