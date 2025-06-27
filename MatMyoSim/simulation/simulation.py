import numpy as np
from MatMyoSim.muscle.muscle import Muscle
import copy

def implement_time_step(myosim_muscle:Muscle, dt, dhsl, pCa, sim_mode, kinetic_scheme):

    n_hs = myosim_muscle.no_of_half_sarcomeres

    myosim_muscle.implement_time_step(dt, dhsl, 10 **(-pCa), sim_mode, kinetic_scheme)

    sim_output = {"muscle_force":myosim_muscle.muscle_force, 
                  "muscle_length":myosim_muscle.muscle_length,
                "command_length": myosim_muscle.command_length,
                "series_extension":myosim_muscle.series_extension,
                "f_overlap": np.full((n_hs), np.nan),
                "f_activated": np.full((n_hs), np.nan),
                "f_bound": np.full((n_hs), np.nan),
                "hs_force": np.full((n_hs), np.nan),
                "cb_force": np.full((n_hs), np.nan),
                "int_pas_force": np.full((n_hs), np.nan),
                "int_total_force": np.full((n_hs), np.nan),
                "ext_pas_force": np.full((n_hs), np.nan),
                "visc_force": np.full((n_hs), np.nan),
                "hs_length": np.full((n_hs), np.nan),
                "Ca": np.full((n_hs), np.nan),
                "r1": np.full((n_hs, len(myosim_muscle.hs[0].rate_structure["r1"])), np.nan),
                "M1": np.full((n_hs), np.nan),
                "M2": np.full((n_hs), np.nan)}

    for i in range(myosim_muscle.no_of_half_sarcomeres):


        sim_output["f_overlap"][i] = myosim_muscle.hs[i].f_overlap
        sim_output["f_activated"][i] = myosim_muscle.hs[i].f_on
        sim_output["f_bound"][i] = myosim_muscle.hs[i].f_bound
        sim_output["hs_force"][i] = myosim_muscle.hs[i].hs_force
        sim_output["cb_force"][i] = myosim_muscle.hs[i].cb_force
        sim_output["int_pas_force"][i] = myosim_muscle.hs[i].int_passive_force
        sim_output["int_total_force"][i] = myosim_muscle.hs[i].int_total_force
        sim_output["ext_pas_force"][i] = myosim_muscle.hs[i].ext_passive_force
        sim_output["visc_force"][i] = myosim_muscle.hs[i].viscous_force
        sim_output["hs_length"][i] = myosim_muscle.hs[i].hs_length
        sim_output["Ca"][i] = myosim_muscle.hs[i].Ca
        sim_output["r1"][i] = myosim_muscle.hs[i].rate_structure["r1"]

        if myosim_muscle.hs[0].kinetic_scheme.startswith("2state"):
            sim_output["M1"][i] = myosim_muscle.hs[i].m_state_pops["M1"]
            sim_output["M2"][i] = myosim_muscle.hs[i].m_state_pops["M2"]
            
            M2_indices = slice(1, 1 + myosim_muscle.hs[i].myofilaments["no_of_x_bins"])
            # sim_output["cb_pops"][i] = myosim_muscle.hs[i].myofilaments["y"][M2_indices]
    
    return myosim_muscle, sim_output

def find_firing_rate(time_step,sim_output_bag, sim_output_chain, force_dynamic_list, yank_list, kFc,kFb,kYb):

    Fs = sim_output_chain['hs_force']
    Fd = sim_output_bag['hs_force']

    if len(force_dynamic_list) == 0:
        Yd = Fd/time_step
    else:
        Yd = (Fd - force_dynamic_list[-1])/time_step
    
    if Yd <0:
        Yd = 0
    
    force_dynamic_list.append(Fd)
    yank_list.append(Yd)

    # print(Fs)
    # print(Fd)
    # print(Yd)

    rs = Fs*kFc
    rd = Fd*kFb + Yd*kYb

    # print(Yd)

    # print(rs)
    # print(rd)

    if rs<0:
        rs = 0
    
    rs = 2*rs/(10**5)
    rd = 2*rd/(10**5)

    r = rs+rd

    if r<0:
        r = 0
    
    return r, rs, rd, force_dynamic_list, yank_list













