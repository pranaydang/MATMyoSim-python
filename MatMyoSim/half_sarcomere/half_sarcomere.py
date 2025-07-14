import numpy as np
import json
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

import numpy as np

class HalfSarcomere:
    def __init__(self, hs_props):#, hs_id):
        #self.hs_id = hs_id

        self.hs_length = hs_props["hs_length"]
        self.command_length = self.hs_length
        self.slack_length = np.nan

        self.hs_force = 0
        self.f_overlap = 0
        self.f_on = 0
        self.f_bound = 0
        self.m_force = 0
        self.c_force = 0
        self.cb_force = 0
        self.int_passive_force = 0
        self.viscous_force = 0
        self.ext_passive_force = 0
        self.int_total_force = 0
        self.check_force = 0
        self.Ca = 0
        self.kinetic_scheme = hs_props["kinetic_scheme"]
        self.parameters = hs_props["parameters"]
        self.myofilaments = hs_props["myofilaments"]
        self.rate_structure = {}
        self.m_state_pops = {}
        self.c_state_pops = {}

        # Set myofilament properties
        mf_props = hs_props["myofilaments"]
        for key, value in mf_props.items():
            self.myofilaments[key] = value

        self.myofilaments["x"] = np.arange(
            self.myofilaments["bin_min"],
            self.myofilaments["bin_max"] + self.myofilaments["bin_width"],
            self.myofilaments["bin_width"]
        )
        self.myofilaments["no_of_x_bins"] = len(self.myofilaments["x"])



        if self.kinetic_scheme.startswith("2state"):
            self.myofilaments["y_length"] = self.myofilaments["no_of_x_bins"] + 3
        elif self.kinetic_scheme.startswith("3state_with_SRX"):
            self.myofilaments["y_length"] = self.myofilaments["no_of_x_bins"] + 4
        elif self.kinetic_scheme.startswith("m_3state_with_SRX_mybpc_2state"):
            self.myofilaments["y_length"] = (2 * self.myofilaments["no_of_x_bins"]) + 5
            self.myofilaments["y"] = np.zeros(self.myofilaments["y_length"])
            self.myofilaments["y"][0] = 1.0
            self.myofilaments["y"][3 + self.myofilaments["no_of_x_bins"]] = 1.0
            self.myofilaments["y"][5 + self.myofilaments["no_of_x_bins"]] = 1.0
        elif self.kinetic_scheme.startswith("4state_with_SRX"):
            self.myofilaments["y_length"] = (2 * self.myofilaments["no_of_x_bins"]) + 4
        elif self.kinetic_scheme.startswith("6state_with_SRX"):
            self.myofilaments["y_length"] = (2 * self.myofilaments["no_of_x_bins"]) + 6
        elif self.kinetic_scheme.startswith("7state_with_SRX"):
            self.myofilaments["y_length"] = (3 * self.myofilaments["no_of_x_bins"]) + 6
        elif self.kinetic_scheme.startswith("beard_atp"):
            self.myofilaments["y_length"] = (4 * self.myofilaments["no_of_x_bins"]) + 5
        elif self.kinetic_scheme == "3D_1A":
            self.myofilaments["y_length"] = self.myofilaments["no_of_x_bins"] + 5
        elif self.kinetic_scheme == "3D_3A":
            self.myofilaments["y_length"] = (3 * self.myofilaments["no_of_x_bins"]) + 5
        elif self.kinetic_scheme == "4D_3A":
            self.myofilaments["y_length"] = (3 * self.myofilaments["no_of_x_bins"]) + 6

        if "y" not in self.myofilaments:
            self.myofilaments["y"] = np.zeros(self.myofilaments["y_length"])
            self.myofilaments["y"][0] = 1.0
            self.myofilaments["y"][-2] = 1.0

        # Load parameter properties
        param_props = hs_props["parameters"]
        default_params = {
            "viscosity": 0,
            "prop_fibrosis": 0,
            "prop_myofilaments": 1,
            "int_passive_force_mode": "linear",
            "int_passive_hsl_slack": 1000,
            "passive_k_linear": 0,
            "ext_passive_force_mode": "linear",
            "ext_passive_hsl_slack": 1000,
            "ext_passive_k_linear": 0,
        }

        for k, v in default_params.items():
            self.parameters[k] = param_props.get(k, v)

        # Forces initialization
        self.cb_force = 0
        self.int_passive_force = self.return_intracellular_passive_force(self.hs_length)
        self.ext_passive_force = self.return_extracellular_passive_force(self.hs_length)
        self.int_total_force = self.cb_force + self.int_passive_force
        self.hs_force = self.int_total_force + self.ext_passive_force

        self.f_on = 0
        self.f_bound = 0
        self.f_overlap = self.return_f_overlap()
    
    def implement_time_step(self, time_step, delta_hsl, Ca_concentration, m_props):

        self.Ca = Ca_concentration

        self.evolve_kinetics(time_step, m_props, delta_hsl)

        if abs(delta_hsl)>0:
            self.move_cb_distribution(delta_hsl)
        
        self.update_forces(time_step, delta_hsl)

        flag = True

        if self.kinetic_scheme.startswith('2state'):
            flag = False
            self.m_state_pops["M1"] = self.myofilaments["y"][0]
            self.m_state_pops["M2"] = sum(self.myofilaments["y"][1 : 1 + self.myofilaments["no_of_x_bins"]])
        
        if flag:
            raise Exception("half_sarcomere::implement_time_step, kinetic scheme undefined")
        
        return None

    def move_cb_distribution(self, delta_hsl):

        delta_x = delta_hsl * self.parameters["compliance_factor"]

        flag = True

        if self.kinetic_scheme.startswith('2state'):
            flag = False
            interp_positions = self.myofilaments["x"] - delta_x
            bin_indices = np.arange(2, self.myofilaments["no_of_x_bins"] + 2)
            cbs_bound_before = sum(self.myofilaments["y"][bin_indices-1])
            interpolator = interp1d(self.myofilaments["x"],self.myofilaments["y"][bin_indices - 1],  kind="linear",fill_value=0,bounds_error=False)
            self.myofilaments["y"][bin_indices-1] = interpolator(interp_positions)
            cbs_lost = cbs_bound_before - sum(self.myofilaments["y"][bin_indices-1])
            self.myofilaments["y"][0] += cbs_lost

        if flag:
            raise Exception("kinetics scheme not yet implemented in move_cb_distributions")
        
        return None
    
    def update_forces(self, time_step, delta_hsl):

        [self.m_force, self.c_force] = self.return_distrib_force(time_step, delta_hsl)
        # print(self.c_force)
        self.int_passive_force = self.return_intracellular_passive_force(self.hs_length)
        self.ext_passive_force = self.return_extracellular_passive_force(self.hs_length)
        # print(self.ext_passive_force)
        self.viscous_force = (1 - self.parameters["prop_fibrosis"]) * self.parameters["prop_myofilaments"] * self.parameters["viscosity"] * delta_hsl / time_step
        # print(self.viscous_force)
        self.int_total_force = self.m_force + self.c_force + self.int_passive_force
        # print(self.m_force)
        self.hs_force = self.m_force + self.c_force + self.int_passive_force + self.viscous_force + self.ext_passive_force #hello hello hello this might be it

        return None
    
    def check_new_force(self, new_length, time_step):

        delta_hs_length = new_length - self.hs_length

        delta_m_force = (1 - self.parameters["prop_fibrosis"]) * self.parameters["prop_myofilaments"] * self.parameters["cb_number_density"] * self.f_bound * self.parameters["compliance_factor"] * delta_hs_length * 1e-9 * self.parameters["k_cb"]

        delta_c_force = 0

        delta_int_pas_force = self.return_intracellular_passive_force(new_length)- self.int_passive_force
        # print(delta_int_pas_force)
        delta_ext_pas_force = self.return_extracellular_passive_force(new_length)- self.ext_passive_force
        # print(delta_ext_pas_force)
        delta_viscous_force = (1 - self.parameters["prop_fibrosis"]) * self.parameters["prop_myofilaments"] * self.parameters["viscosity"] * delta_hs_length / time_step
        # print(delta_viscous_force)
        self.check_force = self.hs_force + delta_m_force + delta_c_force + delta_int_pas_force + delta_ext_pas_force + delta_viscous_force

        return None
    
    def update_2state_with_poly(self, time_step):


        y = self.myofilaments["y"]

        # print(y)

        N_overlap = self.return_f_overlap()

        rate_func = self.parameters["rate_func"]

        # print(rate_func)

        if rate_func == "newSpindleBag1":

            r1 = np.zeros(len(self.myofilaments["x"]))
            r1 = (self.parameters["k_1"]*np.exp(-self.parameters["k_cb"]*10*(self.myofilaments["x"]**2)/ (1e18 * self.parameters["k_boltzmann"] * self.parameters["temperature"])))
            r1[r1>self.parameters["max_rate"]] = self.parameters["max_rate"]

            r2 = np.zeros(len(self.myofilaments["x"]))
            r2[self.myofilaments["x"]<-5] = self.parameters["k_2_0"] + abs(0.02*((self.myofilaments["x"][self.myofilaments["x"]<-5]+5)**3))
            r2[self.myofilaments["x"]>=-5] = self.parameters["k_2_0"] + 0.2*((self.myofilaments["x"][self.myofilaments["x"]>=-5]+5)**3)
            r2 += 0.5
            r2[r2>self.parameters["max_rate"]]= self.parameters["max_rate"]

        elif rate_func == "newSpindleChain1":

            r1 = np.zeros(len(self.myofilaments["x"]))
            r1 = (self.parameters["k_1"]*np.exp(-self.parameters["k_cb"]*5*(2*(self.myofilaments["x"])**2)/ (1e18 * self.parameters["k_boltzmann"] * self.parameters["temperature"])))
            r1[r1>self.parameters["max_rate"]] = self.parameters["max_rate"]

            r2 = np.zeros(len(self.myofilaments["x"]))
            r2[self.myofilaments["x"]<-5] = self.parameters["k_2_0"] + abs(0.2*((self.myofilaments["x"][self.myofilaments["x"]<-5]+5)**3))
            r2[self.myofilaments["x"]>=-5] = self.parameters["k_2_0"] + 0.4*((self.myofilaments["x"][self.myofilaments["x"]>=-5]+5)**3)
            r2 += 10
            r2[r2>self.parameters["max_rate"]]= self.parameters["max_rate"]
        
        def derivs(time_step,y):

            dy = np.zeros_like(y)

            # print(dy)

            # print(y.shape)

            M1 = y[0]
            # print(M1)
            # print(y)
            M2 = y[1 : 1 + self.myofilaments["no_of_x_bins"]]
            # print(len(M2))
            N_off = y[-2]
            N_on = y[-1]
            N_bound = np.sum(M2)

            J1 = r1 * self.myofilaments["bin_width"] * M1 * (N_on - N_bound)
            J2 = r2 * M2.T
            J_on = self.parameters["k_on"] * self.Ca * (N_overlap - N_on) * \
                (1 + self.parameters["k_coop"] * (N_on / N_overlap))

            J_off = self.parameters["k_off"] * (N_on - N_bound) * \
                    (1 + self.parameters["k_coop"] * ((N_overlap - N_on) / N_overlap))


            # Calculate the derivatives
            dy[0] = np.sum(J2) - np.sum(J1)
            for i in range(self.myofilaments["no_of_x_bins"]):
                dy[1 + i] = J1[i] - J2[i]
            dy[-2] = -J_on + J_off
            dy[-1] = J_on - J_off

            return dy
        
        sol = solve_ivp(derivs, [0, time_step], y, method='RK23')

        self.myofilaments["y"] = sol.y[:, -1] 
        self.f_overlap = N_overlap
        self.f_on = self.myofilaments["y"][-1]
        self.f_bound = np.sum(self.myofilaments["y"][1 : 1 + self.myofilaments["no_of_x_bins"]])

        self.rate_structure["r1"] = r1
        self.rate_structure["r2"] = r2

        return None
    
    def evolve_kinetics(self, time_step, m_props, delta_hsl):
        
        kinetic_scheme = self.kinetic_scheme

        if kinetic_scheme == "2state": #it was 2state_with_poly earlier
            self.update_2state_with_poly(time_step)
        
        return None
    
    def return_distrib_force(self, time_step, delta_hsl):

        c_force = 0
        m_force = 0

        kinetic_scheme = self.kinetic_scheme

        if kinetic_scheme == "2state":

            bin_pops = self.myofilaments["y"][1 : 1 + self.myofilaments["no_of_x_bins"]]

            m_force = (
                (1 - self.parameters["prop_fibrosis"])
                * self.parameters["prop_myofilaments"]
                * self.parameters["cb_number_density"]
                * self.parameters["k_cb"]
                * 1e-9
                * np.sum((self.myofilaments["x"] + self.parameters["x_ps"]) * bin_pops)
                )
            
            # print(np.sum((self.myofilaments["x"] + self.parameters["x_ps"]) * bin_pops))
            
            return [m_force, c_force]
    
    def return_intracellular_passive_force(self, hsl):

        mode = self.parameters["int_passive_force_mode"]
        fibrosis_factor = (1 - self.parameters["prop_fibrosis"]) * self.parameters["prop_myofilaments"]

        if mode == "linear":
            pf = fibrosis_factor * self.parameters["int_passive_k_linear"] * (hsl - self.parameters["int_passive_hsl_slack"])

        elif mode == "exponential":
            slack = self.parameters["int_passive_hsl_slack"]
            L = self.parameters["int_passive_L"]
            sigma = self.parameters["int_passive_sigma"]
            delta_hsl = hsl - slack

            if hsl > slack:
                pf = fibrosis_factor * sigma * (np.exp(delta_hsl / L) - 1)
            else:
                pf = fibrosis_factor * -sigma * (np.exp(-delta_hsl / L) - 1)

        else:
            raise ValueError("Passive force mode not defined")

        return pf
    
    def return_extracellular_passive_force(self, hsl):

        mode = self.parameters["ext_passive_force_mode"]

        if mode == "linear":
            pf = self.parameters["prop_fibrosis"] * self.parameters["ext_passive_k_linear"] * (hsl - self.parameters["ext_passive_hsl_slack"])

        elif mode == "exponential":
            if hsl > self.parameters["ext_passive_hsl_slack"]:
                pf = self.parameters["prop_fibrosis"] * self.parameters["ext_passive_sigma"] * (np.exp((hsl - self.parameters["ext_passive_hsl_slack"]) / self.parameters["ext_passive_L"]) - 1)
            else:
                pf = self.parameters["prop_fibrosis"] * -self.parameters["ext_passive_sigma"] * (np.exp(-(hsl - self.parameters["ext_passive_hsl_slack"]) / self.parameters["ext_passive_L"]) - 1)

        else:
            raise ValueError("Passive force mode not defined")

        return pf
    
    def return_f_overlap(self):
        
        x_no_overlap = self.hs_length - self.myofilaments["thick_filament_length"]
        x_overlap = self.myofilaments["thin_filament_length"] - x_no_overlap
        max_x_overlap = self.myofilaments["thick_filament_length"] - self.myofilaments["bare_zone_length"]

        if x_overlap < 0:
            f_overlap = 0
        elif 0 < x_overlap <= max_x_overlap:
            f_overlap = x_overlap / max_x_overlap
        elif x_overlap > max_x_overlap:
            f_overlap = 1

        protrusion = self.myofilaments["thin_filament_length"] - (self.hs_length + self.myofilaments["bare_zone_length"])

        if protrusion > 0:
            x_overlap = max_x_overlap - protrusion
            f_overlap = x_overlap / max_x_overlap

        return f_overlap





            



