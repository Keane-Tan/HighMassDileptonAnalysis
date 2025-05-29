import awkward as ak
import numpy as np
from typing import Union, TypeVar, Tuple
from coffea.nanoevents.methods import vector
from omegaconf import OmegaConf
import logging
from modules.utils import logger

coffea_nanoevent = TypeVar('coffea_nanoevent')
ak_array = TypeVar('ak_array')

def getZptWgts(dimuon_pt, njets, nbins, year, config_path):
    # config_path = "./data/zpt_rewgt/fitting/zpt_rewgt_params.yaml"
    # config_path = config["new_zpt_wgt"]
    logger.info(f"zpt config file: {config_path}")
    wgt_config = OmegaConf.load(config_path)
    max_order = 5 #9
    zpt_wgt = ak.ones_like(dimuon_pt)
    jet_multiplicies = [0,1,2]
    # logger.info(f"zpt_wgt: {zpt_wgt}")

    for jet_multiplicity in jet_multiplicies:

        zpt_wgt_by_jet = ak.zeros_like(dimuon_pt)
        # zpt_wgt_by_jet = ak.ones_like(dimuon_pt) * -1 # debugging
        # polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(max_order+1): # p goes from 0 to max_order
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"p{order}"]
            # logger.info(f"njet{jet_multiplicity} order {order} coeff: {coeff}")
            polynomial_term = coeff*dimuon_pt**order
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term
            # logger.info(f"njet{jet_multiplicity} order {order} polynomial_term: {polynomial_term}")
            # logger.info(f"njet{jet_multiplicity} order {order} zpt_wgt_by_jet_poly: {zpt_wgt_by_jet_poly}")
        poly_fit_cutoff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["x_max"]
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff >= dimuon_pt), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # horizontal line beyond poly_fit_cutoff
        coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"horizontal_c0"]
        zpt_wgt_by_jet_horizontal = ak.ones_like(dimuon_pt) * coeff
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff < dimuon_pt), zpt_wgt_by_jet_horizontal, zpt_wgt_by_jet)
        # logger.info(f"zpt_wgt_by_jet testing: {ak.all(zpt_wgt_by_jet != -1).compute()}")
        # raise ValueError

        if jet_multiplicity != 2:
            njet_mask = njets == jet_multiplicity
        else:
            njet_mask = njets >= 2 # njet 2 is inclusive
        # logger.info(f"njet{jet_multiplicity} order  zpt_wgt_by_jet: {zpt_wgt_by_jet}")
        zpt_wgt = ak.where(njet_mask, zpt_wgt_by_jet, zpt_wgt) # if matching jet multiplicity, apply the values
        # logger.info(f"zpt_wgt after njet {jet_multiplicity}: {zpt_wgt}")

    cutOff_mask = dimuon_pt < 200 # ignore wgts from dimuon pT > 200
    zpt_wgt = ak.where(cutOff_mask, zpt_wgt, ak.ones_like(dimuon_pt))
    return zpt_wgt

def getZptWgts_2016postVFP(dimuon_pt, njets, nbins, year, config_path):
    # config_path = "./data/zpt_rewgt/fitting/zpt_rewgt_params.yaml"
    # config_path = config["new_zpt_wgt"]
    logger.info(f"zpt config file: {config_path}")
    wgt_config = OmegaConf.load(config_path)
    max_order = 5 #9
    zpt_wgt = ak.ones_like(dimuon_pt)
    jet_multiplicies = [0,1,2]
    # logger.info(f"zpt_wgt: {zpt_wgt}")

    for jet_multiplicity in jet_multiplicies:

        zpt_wgt_by_jet = ak.zeros_like(dimuon_pt)
        # zpt_wgt_by_jet = ak.ones_like(dimuon_pt) * -1 # debugging
        # first polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(2+1): # FIXME: Hardcoded polynomial order
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"fp{order}"]
            # logger.info(f"njet{jet_multiplicity} order {order} coeff: {coeff}")
            polynomial_term = coeff*dimuon_pt**order
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term
            # logger.info(f"njet{jet_multiplicity} order {order} polynomial_term: {polynomial_term}")
            # logger.info(f"njet{jet_multiplicity} order {order} zpt_wgt_by_jet_poly: {zpt_wgt_by_jet_poly}")
        poly_fit_cutoff_min = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["x_min"]
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_min >= dimuon_pt), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # polynomial fit
        zpt_wgt_by_jet_poly = ak.zeros_like(dimuon_pt)
        for order in range(max_order+1): # p goes from 0 to max_order
            coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"p{order}"]
            # logger.info(f"njet{jet_multiplicity} order {order} coeff: {coeff}")
            polynomial_term = coeff*dimuon_pt**order
            zpt_wgt_by_jet_poly = zpt_wgt_by_jet_poly + polynomial_term
            # logger.info(f"njet{jet_multiplicity} order {order} polynomial_term: {polynomial_term}")
            # logger.info(f"njet{jet_multiplicity} order {order} zpt_wgt_by_jet_poly: {zpt_wgt_by_jet_poly}")
        poly_fit_cutoff_max = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins]["polynomial_range"]["x_max"]
        zpt_wgt_by_jet = ak.where(((poly_fit_cutoff_min < dimuon_pt) & (poly_fit_cutoff_max >= dimuon_pt)), zpt_wgt_by_jet_poly, zpt_wgt_by_jet)

        # horizontal line beyond poly_fit_cutoff_max
        coeff = wgt_config[str(year)][f"njet_{jet_multiplicity}"][nbins][f"horizontal_c0"]
        zpt_wgt_by_jet_horizontal = ak.ones_like(dimuon_pt) * coeff
        zpt_wgt_by_jet = ak.where((poly_fit_cutoff_max < dimuon_pt), zpt_wgt_by_jet_horizontal, zpt_wgt_by_jet)
        # logger.info(f"zpt_wgt_by_jet testing: {ak.all(zpt_wgt_by_jet != -1).compute()}")
        # raise ValueError

        if jet_multiplicity != 2:
            njet_mask = njets == jet_multiplicity
        else:
            njet_mask = njets >= 2 # njet 2 is inclusive
        # logger.info(f"njet{jet_multiplicity} order  zpt_wgt_by_jet: {zpt_wgt_by_jet}")
        zpt_wgt = ak.where(njet_mask, zpt_wgt_by_jet, zpt_wgt) # if matching jet multiplicity, apply the values
        # logger.info(f"zpt_wgt after njet {jet_multiplicity}: {zpt_wgt}")

    cutOff_mask = dimuon_pt < 200 # ignore wgts from dimuon pT > 200
    zpt_wgt = ak.where(cutOff_mask, zpt_wgt, ak.ones_like(dimuon_pt))
    return zpt_wgt

def merge_zpt_wgt(yun_wgt, valerie_wgt, njets, year):
    """
    helper function that merges yun_wgt and valerie_wgt defined by jet multiplicity
    """
    val_filter_dict_run2 = {
        "2018": {0: False, 1: False, 2: True},
        "2017": {0: True, 1: False, 2: True},
        "2016postVFP": False,
        "2016preVFP": False,
    }
    val_filter_dict = val_filter_dict_run2[year]
    # logger.info(f"val_filter_dict: {val_filter_dict}")
    if val_filter_dict == False:
        return yun_wgt
    else: # divide by njet multiplicity
        val_filter = ak.zeros_like(valerie_wgt, dtype="bool")
        for njet_multiplicity_target, use_flag in val_filter_dict.items():
            # logger.info(f"njet_multiplicity_target: {njet_multiplicity_target}")
            # logger.info(f"{year} njet {njet_multiplicity_target} use_flag: {use_flag}")
            if use_flag == False: # skip
                logger.info("skipping!")
                continue
            # If true, generate a boolean 1-D array
            if njet_multiplicity_target != 2:
                use_valerie_zpt =  njets == njet_multiplicity_target
            else:
                use_valerie_zpt =  njets >= njet_multiplicity_target
            val_filter = val_filter | use_valerie_zpt

            # logger.info(f"{year} njet {njet_multiplicity_target} use_valerie_zpt: {use_valerie_zpt[:20].compute()}")
            # logger.info(f"{year} njet {njet_multiplicity_target} njets: {njets[:20].compute()}")
        # logger.info(f"{year}  val_filter: {val_filter[:20].compute()}")
        # raise ValueError
        final_filter = ak.where(val_filter, valerie_wgt, yun_wgt)
        return final_filter

def getRapidity(obj):
    px = obj.pt * np.cos(obj.phi)
    py = obj.pt * np.sin(obj.phi)
    pz = obj.pt * np.sinh(obj.eta)
    e = np.sqrt(px**2 + py**2 + pz**2 + obj.mass**2)
    rap = 0.5 * np.log((e + pz) / (e - pz))
    return rap

def _mass2_kernel(t, x, y, z):
    return t * t - x * x - y * y - z * z

def p4_sum_mass(obj1, obj2):
    result_px = ak.zeros_like(obj1.pt)
    result_py = ak.zeros_like(obj1.pt)
    result_pz = ak.zeros_like(obj1.pt)
    result_e = ak.zeros_like(obj1.pt)
    for obj in [obj1, obj2]:
        # px_ = obj.pt * np.cos(obj.phi)
        # py_ = obj.pt * np.sin(obj.phi)
        # pz_ = obj.pt * np.sinh(obj.eta)
        px_ = obj.px
        py_ = obj.py
        pz_ = obj.pz
        e_ = np.sqrt(px_**2 + py_**2 + pz_**2 + obj.mass**2)
        result_px = result_px + px_
        result_py = result_py + py_
        result_pz = result_pz + pz_
        result_e = result_e + e_
    # result_pt = np.sqrt(result_px**2 + result_py**2)
    # result_eta = np.arcsinh(result_pz / result_pt)
    # result_phi = np.arctan2(result_py, result_px)
    result_mass = np.sqrt(
        result_e**2 - result_px**2 - result_py**2 - result_pz**2
    )
    result_rap = 0.5 * np.log((result_e + result_pz) / (result_e - result_pz))
    return result_mass

def p4_subtract_pt(obj1, obj2):
    """
    obtain the pt vector subtraction of two variables
    """
    result_px = ak.zeros_like(obj1.pt)
    result_py = ak.zeros_like(obj1.pt)
    result_pz = ak.zeros_like(obj1.pt)
    result_e = ak.zeros_like(obj1.pt)
    coeff = 1.0
    for obj in [obj1, obj2]:
        # px_ = obj.pt * np.cos(obj.phi)
        # py_ = obj.pt * np.sin(obj.phi)
        # pz_ = obj.pt * np.sinh(obj.eta)
        px_ = obj.px
        py_ = obj.py
        pz_ = obj.pz
        result_px = result_px + coeff*px_
        result_py = result_py + coeff*py_
        result_pz = result_pz + coeff*pz_
        coeff = -1*coeff # switch coeff
    result_pt = np.sqrt(result_px**2 + result_py**2)
    return result_pt

def testJetVector(jets):
    """
    This is a helper function in debugging observed inconsistiency in Jet variables after
    migration from coffea native vectors to hep native vectors
    params:
    jets -> nanoevent vector of Jet. IE: events.Jet
    """
    padded_jets = ak.pad_none(jets, target=2)
    # logger.info(f"type padded_jets: {type(padded_jets.compute())}")
    jet1 = padded_jets[:, 0]
    jet2 = padded_jets[:, 1]
    normal_dijet =  jet1 + jet2
    logger.info(f"type normal_dijet: {type(normal_dijet.compute())}")
    # explicitly reinitialize the jets
    jet1_4D_vec = ak.zip({"pt":jet1.pt, "eta":jet1.eta, "phi":jet1.phi, "mass":jet1.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    jet2_4D_vec = ak.zip({"pt":jet2.pt, "eta":jet2.eta, "phi":jet2.phi, "mass":jet2.mass}, with_name="PtEtaPhiMLorentzVector", behavior=vector.behavior)
    new_dijet = jet1_4D_vec + jet2_4D_vec
    target_arr = ak.fill_none(new_dijet.mass.compute(), value=-99.0)
    out_arr = ak.fill_none(normal_dijet.mass.compute(), value=-99.0)
    rel_err = np.abs((target_arr-out_arr)/target_arr)
    logger.info(f"max rel_err: {ak.max(rel_err)}")

def delta_r_V1(eta1, eta2, phi1, phi2):
    deta = abs(eta1 - eta2)
    dphi = abs(np.mod(phi1 - phi2 + np.pi, 2 * np.pi) - np.pi)
    dr = np.sqrt(deta**2 + dphi**2)
    return deta, dphi, dr

def etaFrame_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]:
    """
    Obtain eta frame cos(theta) and phi as specified in:
    https://link.springer.com/article/10.1140/epjc/s10052-011-1600-y and
    This Eta frame values supposedly plays a similar role to CS frame in terms of physics
    sensitivity, but with better resolution. Not nocessarily believe this claim
    however.
    """
    # divide muons in terms of negative and positive charges instead of leading pT
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dphi = abs(mu_neg.delta_phi(mu_pos))
    theta_eta = np.arccos(np.tanh((mu_neg.eta - mu_pos.eta) / 2))
    phi_eta = np.tan((np.pi - np.abs(dphi)) / 2) * np.sin(theta_eta)
    return np.cos(theta_eta), phi_eta

def cs_variables(
        mu1: coffea_nanoevent,
        mu2: coffea_nanoevent
    ) -> Tuple[ak_array]:
    """
    return cos(theta) and phi in collins-soper frame
    """
    dimuon = mu1 + mu2
    cos_theta_cs = getCosThetaCS(mu1, mu2, dimuon)
    phi_cs = getPhiCS(mu1, mu2, dimuon)
    return cos_theta_cs, phi_cs

def getCosThetaCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
    ) -> ak_array :
    """
    return cos(theta) in collins-soper frame
    the formula for cos(theta) is given in Eqn 1. of https://www.ciemat.es/portal.do?TR=A&IDR=1&identificador=813
    """
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    nominator = 2*(mu1.pz*mu2.energy - mu2.pz*mu1.energy)
    demoninator = dimuon_mass * (dimuon_mass**2 + dimuon_pt**2)**(0.5)
    cos_theta_cs = -(nominator/demoninator) # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    return cos_theta_cs

def getPhiCS(
    mu1: coffea_nanoevent,
    mu2: coffea_nanoevent,
    dimuon: coffea_nanoevent,
    ) -> ak_array :
    """
    return phi in collins-soper frame
    the formula for phi is given in Eqn F.8 of https://people.na.infn.it/~elly/TesiAtlas/SpinCP/TestIpotesi/CollinSoperDefinition.pdf
    the implementation is heavily inspired from https://github.com/JanFSchulte/SUSYBSMAnalysis-Zprime2muAnalysis/blob/mini-AOD-2018/src/AsymFunctions.C#L1549-L1603
    """
    mu_neg = ak.where((mu1.charge<0), mu1,mu2)
    mu_pos = ak.where((mu1.charge>0), mu1,mu2)
    dimuon_pz = dimuon.pz
    dimuon_pt = dimuon.pt
    dimuon_mass = dimuon.mass
    beam_vec_z = ak.where((dimuon_pz>0), ak.ones_like(dimuon_pz), -ak.ones_like(dimuon_pz))
    # intialize beam vector as threevector to do cross product
    # beam_vec =  ak.zip(
    #     {
    #         "x": ak.zeros_like(dimuon_pz),
    #         "y": ak.zeros_like(dimuon_pz),
    #         "z": beam_vec_z,
    #     },
    #     with_name="ThreeVector",
    #     behavior=vector.behavior,
    # )
    # logger.info(f"vector.__file__: {vector.__file__}")
    beam_vec =  ak.zip(
        {
            "x": ak.zeros_like(dimuon_pz),
            "y": ak.zeros_like(dimuon_pz),
            "z": beam_vec_z,
        },
        with_name="Momentum3D",
        behavior=vector.behavior
    )
    # apply cross product. note x,y,z of dimuon refers to its momentum, NOT its location
    # mu.px == mu.x, mu.py == mu.y and so on
    dimuon3D_vec = ak.zip({"x":dimuon.x, "y":dimuon.y, "z":dimuon.z}, with_name="Momentum3D", behavior=vector.behavior)
    R_T = beam_vec.cross(dimuon3D_vec) # direct cross product with dimuon doesn't work bc it's a 5D vector with x,y,z,t and charge

    R_T = R_T.unit() # make it a unit vector
    Q_T = dimuon
    Q_coeff = ( ((dimuon_mass*dimuon_mass + (dimuon_pt*dimuon_pt)))**(0.5) )/dimuon_mass
    delta_T_dot_R_T = (mu_neg.px-mu_pos.px)*R_T.x + (mu_neg.py-mu_pos.py)*R_T.y
    delta_R_term = delta_T_dot_R_T
    delta_R_term = -delta_R_term # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_T_dot_Q_T = (mu_neg.px-mu_pos.px)*Q_T.px + (mu_neg.py-mu_pos.py)*Q_T.py
    delta_T_dot_Q_T = -delta_T_dot_Q_T # add negative sign to match the sign on pisa implementation at https://github.com/green-cabbage/copperhead_fork2/blob/Run3/python/math_tools.py#L152-L223
    delta_Q_term = delta_T_dot_Q_T
    delta_Q_term = delta_Q_term / dimuon_pt # normalize since Q_T should techincally be a unit vector
    phi_cs = np.arctan2(Q_coeff*delta_R_term, delta_Q_term)
    return phi_cs
