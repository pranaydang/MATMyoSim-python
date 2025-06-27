import numpy as np
from MatMyoSim.simulation.simulation import implement_time_step, find_firing_rate
from MatMyoSim.muscle.muscle import Muscle
import pandas as pd
import matplotlib.pyplot as plt

def run(bag_model, chain_model, dt, dhsl, pCa, sim_mode, force_dynamic_list, yank_list):


    bag_muscle = Muscle(bag_model)
    chain_muscle = Muscle(chain_model)

    bag_muscle, bag_output = implement_time_step(bag_muscle, dt, dhsl, pCa, sim_mode, kinetic_scheme='2state')

    chain_muscle, chain_output = implement_time_step(chain_muscle, dt, dhsl, pCa, sim_mode, kinetic_scheme='2state')

    r, rs, rd, force_dynamic_list, yank_list = find_firing_rate(0.001,bag_output,chain_output,force_dynamic_list,yank_list,kFc=0.5,kFb=0.4,kYb=0.005)

    return r, rs, rd, force_dynamic_list, yank_list

if __name__ == "__main__":

    bag_model = r"C:\Users\prana\OneDrive\Desktop\Simha 2023\MatMyoSim\models\2StateBag.json"
    chain_model = r"C:\Users\prana\OneDrive\Desktop\Simha 2023\MatMyoSim\models\2StateChain.json"

    df = pd.read_csv("sineSim48pCa64Freq1Amp16000.txt", sep="\t")

    dhsl = df["dhsl"].values
    dt = df["dt"].values
    mode = df["Mode"].values*2
    pCa = df["pCa"].values

    # print(mode)

    r_total = []
    force_dynamic_list = []
    yank_list = []
    rss = []
    rds = []

    for i in range(len(dhsl)):

        r, rs, rd, force_dynamic_list, yank_list = run(bag_model, chain_model, dt[i], dhsl[i], pCa[i], mode[i], force_dynamic_list, yank_list)
        r_total.append(r)
        rss.append(rs)
        rds.append(rd)
        # print(r)

    time_steps = [i for i in range(1,len(dt)+1)]
    plt.plot(time_steps,r_total)
    plt.plot(time_steps,rss)
    plt.plot(time_steps,rds)
    plt.show()

