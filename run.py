import numpy as np
from MatMyoSim.simulation.simulation import implement_time_step, find_firing_rate, find_firing_rate_matlab
from MatMyoSim.muscle.muscle import Muscle
import pandas as pd
import matplotlib.pyplot as plt

def run(bag_muscle, chain_muscle, dt, dhsl, pCa, sim_mode, force_dynamic_list, yank_list):

    # print(dhsl)

    bag_muscle, bag_output = implement_time_step(bag_muscle, dt, dhsl, pCa, sim_mode, kinetic_scheme='2state')

    chain_muscle, chain_output = implement_time_step(chain_muscle, dt, dhsl, pCa, sim_mode, kinetic_scheme='2state')

    r, rs, rd, force_dynamic_list, yank_list = find_firing_rate(dt,bag_output,chain_output,force_dynamic_list,yank_list,kFc=0.5,kFb=0.4,kYb=0.005)

    # force_dynamic_list, force_static_list = find_firing_rate_matlab(dt,bag_output, chain_output, force_dynamic_list, force_static_list)
    # print(bag_muscle.muscle_length)
    # print(r)

    return r, rs, rd, force_dynamic_list, yank_list, bag_muscle, chain_muscle

    # return force_dynamic_list, force_static_list

if __name__ == "__main__":

    bag_model = r"C:\Users\prana\OneDrive\Desktop\Simha 2023\MatMyoSim\models\2StateBag.json"
    chain_model = r"C:\Users\prana\OneDrive\Desktop\Simha 2023\MatMyoSim\models\2StateChain.json"

    df = pd.read_csv("sineSim48pCa64Freq1Amp16000.txt",sep="\t")

    dhsl = df["dhsl"].values
    dt = df["dt"].values
    mode = df["Mode"].values*2
    print(mode[0])
    pCa = df["pCa"].values

    bag_muscle = Muscle(bag_model)
    chain_muscle = Muscle(chain_model)

    # print(mode)

    r_total = []
    force_dynamic_list = []
    force_static_list = []
    yank_list = []
    rss = []
    rds = []

    for i in range(len(dhsl)):

        r, rs, rd, force_dynamic_list, yank_list, bag_muscle, chain_muscle = run(bag_muscle, chain_muscle, dt[i], dhsl[i], pCa[i], mode[i], force_dynamic_list, yank_list)
        r_total.append(r)
        rss.append(rs)
        rds.append(rd)
        # print(bag_muscle.muscle_length)
        # print(i)
        # if i>3900 and i<3910:
        #     print("about to happen")
        # force_dynamic_list, force_static_list = run(bag_muscle, chain_muscle, dt[i], dhsl[i], pCa[i], mode[i], force_dynamic_list, force_static_list, yank_list)
    
    # yank = np.diff(force_dynamic_list)/dt[0]
    # yank = np.append(yank, np.array([0]))
    # yank[yank < 0] = 0 

    # r_s = 0.5*force_static_list
    # r_d = 0.4*force_dynamic_list + 0.005*yank

    # r_s[r_s<0] = 0

    # r_total = r_d+r_s

    # r_total[r_total<0] = 0

       
    time_steps = [i for i in range(1,len(dt)+1)]
    plt.plot(time_steps,r_total)
    plt.plot(time_steps,rss)
    plt.plot(time_steps,rds)
    # plt.plot(time_steps,r_s)
    # plt.plot(time_steps,r_d)
    plt.show()

