import numpy as np
import json
from scipy.optimize import fsolve
from MatMyoSim.half_sarcomere.half_sarcomere import HalfSarcomere

class Muscle:
    def __init__(self, model_file_path):
        # Load and unpack model
        model = json.load(open(model_file_path))
        myosim = model["MyoSim_model"]
        self.muscle_props = myosim["muscle_props"]
        self.hs_props = myosim["hs_props"]
        self.series_extension = 0
        self.muscle_force = 0


        # Set number of half-sarcomeres
        self.no_of_half_sarcomeres = self.muscle_props["no_of_half_sarcomeres"]

        # Correct series stiffness
        self.series_k_linear = self.muscle_props["series_k_linear_per_hs"] / self.no_of_half_sarcomeres

        # Create half-sarcomeres
        self.hs:list[HalfSarcomere] = []
        self.muscle_length = 0
        for i in range(self.no_of_half_sarcomeres):
            hs = HalfSarcomere(self.hs_props.copy())
            self.hs.append(hs)
            self.muscle_length += hs.hs_length

        self.command_length = self.muscle_length

        # Handle heterogeneity
        if "hs_heterogeneity" in myosim:
            for field, values in myosim["hs_heterogeneity"].items():
                for i in range(self.no_of_half_sarcomeres):
                    self.hs[i].params[field] = values[i]

        # Run equilibrium dummy step
        self.implement_time_step(time_step=-1, delta_hsl=0, Ca_value=10**-9.0, mode_value=-2, kinetic_scheme='2state')

    def implement_time_step(self,time_step,delta_hsl,Ca_value,mode_value, kinetic_scheme):

        self.command_length += self.no_of_half_sarcomeres*delta_hsl

        # print(delta_hsl)
        # print(time_step)
        # print(mode_value)
        # print(self.series_k_linear)
        # print(self.no_of_half_sarcomeres)

        if (self.series_k_linear>0) or (self.no_of_half_sarcomeres > 1) or (mode_value >= 0):

            m_props = {}

            if time_step>0:
                for hs_counter in range(self.no_of_half_sarcomeres):
                    self.hs[hs_counter].implement_time_step(time_step, 0 , Ca_value, m_props)
                
            if mode_value<0:
                self.muscle_length += self.no_of_half_sarcomeres*delta_hsl
                dhsl_balance = self.return_delta_hsl_for_force_balance(mode_value,time_step)
                # print(dhsl_balance)
                for hs_counter in range(self.no_of_half_sarcomeres):
                    self.hs[hs_counter].move_cb_distribution(dhsl_balance[hs_counter])
                    self.hs[hs_counter].hs_length += dhsl_balance[hs_counter]
                    self.hs[hs_counter].update_forces(time_step, 0)
                
                self.muscle_force = self.hs[0].hs_force

                if self.series_k_linear > 0:
                    self.series_extension = self.return_series_extension(self.muscle_force)
                else:
                    self.series_extension = 0
            
            else:
                dhsl_balance = self.return_delta_hsl_for_force_balance(mode_value, time_step)

                self.muscle_length = 0
                for hs_counter in range(self.no_of_half_sarcomeres):
                    self.hs[hs_counter].move_cb_distribution(dhsl_balance[hs_counter])
                    self.hs[hs_counter].hs_length += dhsl_balance[hs_counter]
                    self.hs[hs_counter].update_forces(time_step, 0)
                    self.muscle_length += self.hs[hs_counter].hs_length
                
                self.muscle_force = self.hs[0].hs_force
                if (self.series_k_linear > 0):
                    self.series_extension = self.return_series_extension(self.muscle_force)
                else:
                    self.series_extension = 0
                
                self.muscle_length += self.series_extension
            
        else:
            
            m_props = {}
            self.series_extension = 0
            
            if (mode_value == -1):
                if (time_step > 0):
                    self.hs[0].implement_time_step(time_step, 0, Ca_value, m_props)
                
                isotonic_force = 0
                def tension_control_single_half_sarcomere(p):
                    self.hs[0].check_new_force(p, time_step)
                    return self.hs[0].check_force - isotonic_force
                
                self.hs[0].slack_length = fsolve(tension_control_single_half_sarcomere, x0=0.0)[0]
                
                new_length = max(self.hs[0].slack_length, self.command_length)
                
                adjustment = new_length - self.hs[0].hs_length

                self.hs[0].move_cb_distribution(adjustment)
                self.hs[0].hs_length = new_length
                # self.hs[0].update_forces(time_step, 0)
            
            else:
                if (time_step > 0):
                    self.hs[0].implement_time_step(time_step, delta_hsl, Ca_value, m_props)
                
                self.hs[0].hs_length = self.hs[0].hs_length + delta_hsl

                # print("hello")
            
            self.muscle_length = self.hs[0].hs_length
            self.muscle_force = self.hs[0].hs_force
        
        return None

    def return_delta_hsl_for_force_balance(self ,mode_value, time_step):

        def length_control_muscle_system(p):

            x = np.zeros(len(p))
            # x = x.reshape(-1, 1)

            for i in range(self.no_of_half_sarcomeres):
                self.hs[i].check_new_force(p[i], time_step)
                x[i] = self.hs[i].check_force - p[-1]
            
            if self.series_k_linear>0:
                new_series_extension = self.muscle_length-sum(p[:self.no_of_half_sarcomeres])
                x[-1] = self.return_series_force(new_series_extension)-p[-1]
            
            else:
                x[-1] = self.muscle_length - sum(p[:-1])
            
            return x
        
        def tension_control_muscle_system(p):

            x = np.zeros(len(p))
            # x = x.reshape(-1, 1)

            for i in range(self.no_of_half_sarcomeres):
                self.hs[i].check_new_force(p[i], time_step)
                x[i] = self.hs[i].check_force - mode_value
            
            x[-1] = self.return_series_force(p[-1]) - mode_value

            return x
        
        def tension_control_single_half_sarcomere(p):

            self.hs[0].check_new_force(p, time_step)
            x = self.hs[0].check_force - mode_value

            return x
        
        if (self.no_of_half_sarcomeres==1) and (self.series_k_linear == 0):

            self.series_extension = 0
            
            p = self.hs[0].hs_length
            
            new_p = fsolve(tension_control_single_half_sarcomere, p)
            
            delta_hsl = new_p - p

            # print(delta_hsl)
            
            return delta_hsl
        
        if mode_value<0:

            p = np.zeros(self.no_of_half_sarcomeres + 1)

            for hs_counter in range(self.no_of_half_sarcomeres):
                p[hs_counter] = self.hs[hs_counter].hs_length
            
            p[self.no_of_half_sarcomeres] = self.muscle_force

            new_p = fsolve(length_control_muscle_system, p)

            delta_hsl = new_p - p

            return delta_hsl
        
        else:

            p = np.zeros(self.no_of_half_sarcomeres + 1)

            for hs_counter in range(self.no_of_half_sarcomeres):
                p[hs_counter] = self.hs[hs_counter].hs_length
            
            p[self.no_of_half_sarcomeres] = self.series_extension

            new_p = fsolve(tension_control_muscle_system, p)

            delta_hsl = new_p - p

            return delta_hsl

    def return_series_extension(self, muscle_force):

        return muscle_force / self.series_k_linear
    
    def return_series_force(self, series_extension):

        return self.series_k_linear*series_extension
    




        
            

                
        

