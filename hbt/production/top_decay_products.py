# coding: utf-8

"""
Producers that determine the generator-level particles related to a top quark decay.
"""
import numpy as np
from columnflow.production import Producer, producer
from columnflow.util import maybe_import
from columnflow.columnar_util import set_ak_column
from columnflow.production.util import attach_coffea_behavior
from columnflow.columnar_util import EMPTY_FLOAT, EMPTY_INT
from functools import partial

ak = maybe_import("awkward")


@producer(
    uses={"GenPart.*", attach_coffea_behavior},
    produces={"top_family.*"},  # "reco_top_mass", "top_mass"},
)
def top_decay_products(self: Producer, events: ak.Array, **kwargs) -> ak.Array:
    """
    Creates a new ragged column "top_family" that stores the tops and their decay products. The structure
    will be as follows:

    .. code-block:: python

        [
            # event 1
            [
                [t1,t2], [b_t1, b_t2], [W_t1, W_t2], [q_1, q_2], [qbar_1, qbar_2], [l_1, l_2], [lbar_1, lbar_2],
                [nu_1, nu_2], [nubar_1, nubar_2]
            ],
            # event 2
            ...
        ],

    where the first entry in each array belongs to the first top, and the second entry in each array belongs to the
    second top.
    """
    from IPython import embed
    embed(header="top_decay_products")

    # find hard top quarks
    mother_gen_flags = ["isLastCopy", "fromHardProcess"]
    children_gen_flags = ["isFirstCopy", "fromHardProcess"]

    abs_id = abs(events.GenPart.pdgId)
    tops = events.GenPart[abs_id == 6]
    tops = tops[tops.hasFlags(*children_gen_flags)]

    # distinct top quark children (b's and W's)
    tops_children = tops.distinctChildrenDeep
    tops_children = tops_children[tops_children.hasFlags(*children_gen_flags)]
    tops = events.GenPart[abs_id == 6]
    tops = tops[tops.hasFlags(*mother_gen_flags)]

    # sort different decay products
    # making this very ugly:
    bottoms = tops_children[abs(tops_children.pdgId) == 5]
    bottoms = ak.flatten(bottoms, axis=2)
    w_bosons = tops_children[abs(tops_children.pdgId) == 24]
    w_children = w_bosons.distinctChildrenDeep
    w_children = w_children[w_children.hasFlags(*children_gen_flags)]
    w_bosons = ak.flatten(w_bosons, axis=2)
    qq = w_children[w_children.pdgId > 0]
    qq = qq[qq.pdgId < 5]
    qq = ak.flatten(qq, axis=3)
    qq = ak.firsts(qq, axis=2)                  # ist das korrekt so? siehe higgs beispiel
    # qq = ak.flatten(qq, axis=2)
    qbarqbar = w_children[w_children.pdgId < 0]
    qbarqbar = qbarqbar[qbarqbar.pdgId > -5]
    qbarqbar = ak.flatten(qbarqbar, axis=3)
    qbarqbar = ak.firsts(qbarqbar, axis=2)
    # qbarqbar = ak.flatten(qbarqbar, axis=2)

    leps = ak.concatenate([
        w_children[w_children.pdgId == 11], w_children[w_children.pdgId == 13], w_children[w_children.pdgId == 15],
    ], axis=3)
    leps = ak.flatten(leps, axis=3)
    leps = ak.firsts(leps, axis=2)

    antileps = ak.concatenate([
        w_children[w_children.pdgId == -11], w_children[w_children.pdgId == -13], w_children[w_children.pdgId == -15],
    ], axis=3)
    antileps = ak.flatten(antileps, axis=3)
    antileps = ak.firsts(antileps, axis=2)

    neutrinos = ak.concatenate([
        w_children[w_children.pdgId == 12], w_children[w_children.pdgId == 14], w_children[w_children.pdgId == 16],
    ], axis=3)
    neutrinos = ak.flatten(neutrinos, axis=3)
    neutrinos = ak.firsts(neutrinos, axis=2)

    antineutrinos = ak.concatenate([
        w_children[w_children.pdgId == -12], w_children[w_children.pdgId == -14], w_children[w_children.pdgId == -16],
    ], axis=3)
    antineutrinos = ak.flatten(antineutrinos, axis=3)
    antineutrinos = ak.firsts(antineutrinos, axis=2)

    other_w_children = w_bosons.distinctChildrenDeep[abs(w_bosons.distinctChildrenDeep.pdgId) > 18]
    other_w_children = other_w_children[other_w_children.hasFlags("isFirstCopy")]
    other_w_children = ak.firsts(other_w_children, axis=2)

    # build the column
    top_family = ak.concatenate([
        tops[:, None, :],
        bottoms[:, None, :],
        w_bosons[:, None, :],
        qq[:, None, :],
        qbarqbar[:, None, :],
        leps[:, None, :],
        antileps[:, None, :],
        neutrinos[:, None, :],
        antineutrinos[:, None, :],
        other_w_children[:, None, :],
    ], axis=1)

    # save the column: Keep Nones from being saved
    set_ak_column_f32 = partial(set_ak_column, value_type=np.float32)
    set_ak_column_i32 = partial(set_ak_column, value_type=np.int32)

    float_fields = ("eta", "mass", "phi", "pt", "vx", "vy", "vz", "iso")
    for field in top_family.fields:
        if field in float_fields:
            events = set_ak_column_f32(events, f"top_family.{field}", ak.fill_none(top_family[field], EMPTY_FLOAT))
        else:
            events = set_ak_column_i32(events, f"top_family.{field}", ak.fill_none(top_family[field], EMPTY_INT))

    return events


# @top_decay_products.skip
# def gen_top_decay_products_skip(self: Producer) -> bool:
#     """
#     Custom skip function that checks whether the dataset is a MC simulation containing top
#     quarks in the first place.
#     """
#     # never skip when there is not dataset
#     if not getattr(self, "dataset_inst", None):
#         return False

#     return self.dataset_inst.is_data or not self.dataset_inst.has_tag("has_top")
