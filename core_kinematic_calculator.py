# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:56:36 2021
@author: vargh
"""

import numpy as np
import pandas as pd
from sympy import symbols, pi, Eq, integrate, diff, init_printing, solve
from scipy.optimize import curve_fit
from scipy.integrate import cumtrapz
from scipy.interpolate import interp1d

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon

import matplotlib.pyplot as plt
from tqdm import tqdm

#init_printing()

## Functions
def calc_geom(threeD_balloon, theta):
    norm_vec = np.array([np.cos(theta), 0, np.sin(theta)])
    proj_of_u_on_n = (np.dot(threeD_balloon, norm_vec))*norm_vec.reshape(len(norm_vec), 1)
    proj_of_u_on_n = threeD_balloon - proj_of_u_on_n.transpose()
        
    points = np.zeros((threeD_balloon.shape[0], 2))
    points[:, 0] = proj_of_u_on_n[:, 1]
    points[:, 1] = proj_of_u_on_n[:, 2]
        
    hull = ConvexHull(points) 
    bound = points[hull.vertices]
    perp_A_x = Polygon(bound).area
    
    cent_y = Polygon(bound).centroid.coords[0][0]
    
    norm_vec2 = np.array([np.sin(theta), 0, np.cos(theta)])
    proj_of_u_on_n2 = (np.dot(threeD_balloon, norm_vec2))*norm_vec2.reshape(len(norm_vec2), 1)
    proj_of_u_on_n2 = threeD_balloon - proj_of_u_on_n2.transpose()
        
    points2 = np.zeros((threeD_balloon.shape[0], 2))
    points2[:, 0] = proj_of_u_on_n2[:, 0]
    points2[:, 1] = proj_of_u_on_n2[:, 1]
        
    hull2 = ConvexHull(points2) 
    bound2 = points2[hull2.vertices]
    perp_A_y = Polygon(bound2).area
    cent_x = Polygon(bound2).centroid.coords[0][0]
    
    return perp_A_x, perp_A_y, cent_x, cent_y

def init_calc(threeD_balloon, payload_height, payload_width, payload_depth, connector_height, balloon_height, balloon_mass, COG_payload_h, COG_payload_w, rho_atmo, dim_scale, dyn_visc, F_b, thrust_f, thrust_r, m_dot_f, m_dot_r, acc_g, consider_bouyancy_drift, time_step, target_range, d_tol, dragthrustratio, min_burn_index, moment_arm_thruster):
    ## Initializations
    t = np.array([0]) # time
    r_m = np.array([total_rover_mass]) # full rover mass
    
    # kinematics in x, displacement, velocity and acceleration
    d_x = np.array([0]) # (m)
    v_x = np.array([0]) # (m/s)
    a_x = np.array([0]) # (m/s^2)
    
    # kinematics in y, displacement, velocity and acceleration
    d_y = np.array([0]) # (m)
    v_y = np.array([0]) # (m/s)
    a_y = np.array([0]) # (m/s^2)
    
    # moment about z
    m_z = np.array([0]) # (Nm)
    
    F = np.array([thrust_f]) # Thrust (N)
    D_x = np.array([0]) # Drag in x (N)
    D_y = np.array([0]) # Drag in y (N)
    
    # rotational kinematics in z, displacement, velocity, accleration
    alpha = np.array([0]) # (rad/s^2)
    omega = np.array([0]) # (rad/s)
    theta = np.array([0]) # (rad)
    
    rem_fuel = np.array([fuel_mass])
    
    ballast_mass = np.array([0]) 
    
    i = 0
    fail = 0
    burn_index = 0
    
    while abs(d_x[i] - target_range) > d_tol and not(fail == 1):
        ##  initial conditions
        prev_t = t[i]
        prev_r_m = r_m[i]
        
        prev_d_x = d_x[i]
        prev_v_x = v_x[i]
        prev_a_x = a_x[i]
        
        prev_d_y = d_y[i]
        prev_v_y = v_y[i]
        prev_a_y = a_y[i]
        
        prev_m_z = m_z[i]
        
        prev_F = F[i]
        prev_D_x = D_x[i]
        prev_D_y = D_y[i]
        
        prev_alpha = alpha[i]
        prev_omega = omega[i]
        prev_theta = theta[i]
        
        prev_fuel = rem_fuel[i]
        
        prev_ballast_mass = ballast_mass[i]
        
        ## time
        t = np.append(t, prev_t + time_step)
        cur_t = prev_t + time_step
        
        ## Modified perpendicular area
        perp_A_x, perp_A_y, cent_x, cent_y = calc_geom(threeD_balloon, prev_theta) # calculates perpendicular area in x and y and the centroid for a given theta
        
        ## Center of Gravity, Center of Drag, Moment of Inertia (not rotated)
        COG_balloon_h = (payload_height + connector_height + balloon_height/2)
        COG_balloon_w = cent_x
        
        COG_cur_h = ((r_m[i] - balloon_mass)*COG_payload_h + balloon_mass*COG_balloon_h)/(r_m[i]) # calculates changing height COG
        COG_cur_w = ((r_m[i] - balloon_mass)*COG_payload_w + balloon_mass*COG_balloon_w)/(r_m[i]) # calculates changing COG
        
        J_payload_u = r_m[i]*(payload_height**2 + payload_width**2) # untransformed moment of inertia of payload
        trans_payload_J_d = np.sqrt(COG_cur_h**2 + COG_cur_w**2) - COG_payload # distance axis of rotation must be moved
        J_payload_t = J_payload_u + r_m[i]*trans_payload_J_d**2 # moving axis of rotation with parallel axis theorem
            
        trans_balloon_J_d = np.sqrt((COG_balloon_h - COG_cur_h)**2 + (COG_balloon_w - COG_cur_w)**2) # distance axis of rotation must be moved
        J_balloon_t = J_balloon_u + balloon_mass*trans_balloon_J_d**2 # moving axis of rotation with parallel axis theorem
            
        J_tot = J_payload_t + J_balloon_t    
        
        COD_balloon_h = COG_balloon_h # needs to be updated based on CFD
        COD_balloon_w = COG_balloon_w # needs to be updated based on CFD
        
        # Skin Friction coefficient
        if prev_v_x != 0:
            re_num = rho_atmo*prev_v_x*dim_scale/dyn_visc # Reynold's Number
            C_f = .027/np.power(re_num, 1/7) ## Prandtl's 1/7 Power Law
        else:
            C_f = 0 # If velocity = 0, C_f = 0
        
        D_mag = np.sqrt(prev_D_x**2 + prev_D_y**2) # magnitude of drag
        
        res_freq = int(np.ceil(2*pi*np.sqrt(J_tot/(F_b*balloon_height)))) # calculated resonant frequency
        
        thrust = thrust_f # thrust
        m_dot = m_dot_f # mass flow rate
            
        if abs(D_mag/thrust) < dragthrustratio: # if thrust to drag ratio is less than max ratio, burn
            burn_condition = 1
        else:
            if burn_index > min_burn_index: # if engine has burned for minimal time, and drag condition exceeded, stop burning
                burn_condition = 0    
                burn_index = 0 
        
        if burn_condition:
            burn_index = burn_index + 1
            
            ## Force
            cur_F = thrust
            
            cur_fuel = prev_fuel - m_dot*time_step
            # Ballast
            cur_ballast_mass = prev_ballast_mass + m_dot*time_step
            
            cur_r_m = prev_r_m
            
        else:
            cur_F = 0
            cur_r_m = prev_r_m
            
            cur_fuel = prev_fuel
            
            mass_deficit = 0
            cur_ballast_mass = prev_ballast_mass
            
        perp_A_pay_x = payload_width/np.cos(prev_theta)*payload_depth # calculates perpendicular surface area of payload
            
        pay_drag_x = -.5*(C_D_payload+C_f)*perp_A_pay_x*rho_atmo*prev_v_x**2 # calculates drag from payload 
                
        ball_drag_x = -.5*(C_D_balloon+C_f)*perp_A_x*rho_atmo*prev_v_x**2 # calculates drag from balloon in x
        ball_drag_y = -.5*(C_D_balloon+C_f)*perp_A_y*rho_atmo*prev_v_y**2 # calculates drag from balloon in y
        
        cur_D_x = pay_drag_x + ball_drag_x # calculates total drag in x
        cur_D_y =  ball_drag_y # calculates total drag in y
        cur_D_mag = np.sqrt(cur_D_x**2 + cur_D_y**2) # Magnitude of drag
        
        ## Linear Kinematics
        tot_force_x = cur_F*np.cos(prev_theta) + cur_D_x # effective thrust in x
        tot_force_y = cur_F*np.sin(prev_theta) + cur_D_y # effective force in y
              
        
        
        cur_a_x = tot_force_x/cur_r_m
        cur_a_y = tot_force_y/cur_r_m
        
        cur_v_x = prev_v_x+cur_a_x*time_step
        cur_v_y = prev_v_y+cur_a_y*time_step
        
        cur_d_x = prev_d_x+cur_v_x*time_step
        cur_d_y = prev_d_y+cur_v_y*time_step
        
        
        ## Rotational Kinematics
        # Payload Gravity Torque
        g_m_a_y_pay = COG_cur_h - COG_payload_h # moment arm for gravity on the payload y
        g_m_a_x_pay = COG_cur_w - COG_payload_w # moment arm for gravity on the payload x
        g_m_a_pay = np.sqrt(g_m_a_y_pay**2 + g_m_a_x_pay**2)
        g_m_pay = abs((cur_r_m - balloon_mass)*acc_g * np.sin(prev_theta) * g_m_a_pay)
        
        # Balloon Gravity Torque
        g_m_a_y_ball = COG_cur_h - COG_balloon_h # moment arm for gravity on the payload y
        g_m_a_x_ball = COG_cur_w - COG_balloon_w # moment arm for gravity on the payload x
        g_m_a_ball = np.sqrt(g_m_a_y_pay**2 + g_m_a_x_pay**2)
        g_m_ball = -abs((cur_r_m - balloon_mass)*acc_g * np.sin(prev_theta) * g_m_a_ball)
        
        g_m = g_m_pay + g_m_ball
        
        # Balloon Drag Torque
        d_m_a_y = COD_balloon_h - COG_cur_h # moment arm for drag on the balloon y
        d_m_a_x = COD_balloon_w - COG_cur_w # moment arm for drag on the balloon x
        d_m_a = np.sqrt(d_m_a_y**2 + d_m_a_x**2) # euclidean distance
        
        ball_D_mag = np.sqrt(ball_drag_x**2 + ball_drag_y**2) # magnitude of drag on balloon
        
        d_m = d_m_a*ball_D_mag*np.cos(prev_theta) - pay_drag_x*g_m_a_pay # sum all drag moments
        
        # Bouyancy force torque, balloon
        b_m_a_y = COG_balloon_h - COG_cur_h # moment arm for bouyancy force y 
        b_m_a_x = COG_balloon_w - COG_cur_w # moment arm for bouyancy force x
        b_m_a = np.sqrt(b_m_a_y**2 + b_m_a_x**2) # euclidean 
        
        b_m = b_m_a * F_b * np.sin(prev_theta) # total buoyancy moment
        
        t_m_a = moment_arm_thruster # thruster moment arm
        t_m = cur_F * (moment_arm_thruster) # thruster moment
        
        m_z_tot = d_m - b_m + t_m - g_m # total moment
        cur_alpha = m_z_tot / J_tot
        cur_omega = prev_omega + cur_alpha*time_step
        cur_theta = prev_theta + cur_omega*time_step
        
        ## all updates
        F = np.append(F, cur_F)
        r_m = np.append(r_m, cur_r_m)
        
        D_x = np.append(D_x, cur_D_x)
        D_y = np.append(D_y, cur_D_y)  
        
        a_x = np.append(a_x, cur_a_x)
        a_y = np.append(a_y, cur_a_y)
        
        v_x = np.append(v_x, cur_v_x)
        v_y = np.append(v_y, cur_v_y)
            
        d_x = np.append(d_x, cur_d_x)
        d_y = np.append(d_y, cur_d_y)
        
        m_z = np.append(m_z, m_z_tot)
        alpha = np.append(alpha, cur_alpha)
        omega = np.append(omega, cur_omega)
        theta = np.append(theta, cur_theta) 
        
        rem_fuel = np.append(rem_fuel, cur_fuel)
        
        ballast_mass = np.append(ballast_mass, cur_ballast_mass)
        
        i = i + 1
        if cur_fuel < 0:
            fail = 1
            print('Not Enough Fuel Mass')
            
        if i % 100 == 0:
            print('.', end= '')
            
        if i % 5000 == 0:
            print('\n')
        
    all_data = np.zeros((len(t), 17))
    
    all_data[:, 0] = t
    all_data[:, 1] = F
    all_data[:, 2] = r_m
    all_data[:, 3] = D_x
    all_data[:, 4] = D_y
    all_data[:, 5] = a_x
    all_data[:, 6] = a_y
    all_data[:, 7] = v_x
    all_data[:, 8] = v_y
    all_data[:, 9] = d_x
    all_data[:, 10] = d_y
    all_data[:, 11] = m_z
    all_data[:, 12] = alpha
    all_data[:, 13] = omega
    all_data[:, 14] = theta
    all_data[:, 15] = rem_fuel
    all_data[:, 16] = ballast_mass
    
    headers = ['time', 'force', 'mass', 'drag_x', 'drag_y', 'acceleration_x', 'acceleration_y', 'velocity_x', 'velocity_y', 'displacement_x', 'displacement_y', 'moment_z', 'alpha', 'omega', 'theta', 'fuel_mass', 'ballast_mass']
    return pd.DataFrame(all_data, columns=headers)

def drag_stop_calc(test, ind_ignore, maneuver_time, max_vel, forward_burn_frac, ind_at_end, threeD_balloon, payload_height, payload_width, payload_depth, connector_height, balloon_height, balloon_mass, COG_payload_h, COG_payload_w, rho_atmo, dim_scale, dyn_visc, F_b, thrust_f, thrust_r, m_dot_f, m_dot_r, acc_g, consider_bouyancy_drift, time_step, target_range, d_tol, dragthrustratio, min_burn_index, moment_arm_thruster):    
    ## Drag Stop
    
    reverse_burn_frac = 1 - forward_burn_frac # deprecated if no reverse burn
    
    cutoff_time = maneuver_time * forward_burn_frac

    ## Initializations
    t = np.array([0]) # time
    r_m = np.array([total_rover_mass]) # full rover mass
    
    # kinematics in x, displacement, velocity and acceleration
    d_x = np.array([0]) # (m)
    v_x = np.array([0]) # (m/s)
    a_x = np.array([0]) # (m/s^2)
    
    # kinematics in y, displacement, velocity and acceleration
    d_y = np.array([0]) # (m)
    v_y = np.array([0]) # (m/s)
    a_y = np.array([0]) # (m/s^2)
    
    # moment about z
    m_z = np.array([0]) # (Nm)
    
    F = np.array([thrust_f]) # Thrust (N)
    D_x = np.array([0]) # Drag in x (N)
    D_y = np.array([0]) # Drag in y (N)
    
    # rotational kinematics in z, displacement, velocity, accleration
    alpha = np.array([0]) # (rad/s^2)
    omega = np.array([0]) # (rad/s)
    theta = np.array([0]) # (rad) 
    
    rem_fuel = np.array([fuel_mass])
    
    ballast_mass = np.array([0]) 
    
    i = 0
    fail = 0
    burn_index = 0
    vel_checker = 10 # lets loop accelerate the craft
    while vel_checker >= vel_elbow and not(fail == 1):
        ##  initial conditions
        prev_t = t[i]
        prev_r_m = r_m[i]
        
        prev_d_x = d_x[i]
        prev_v_x = v_x[i]
        prev_a_x = a_x[i]
        
        prev_d_y = d_y[i]
        prev_v_y = v_y[i]
        prev_a_y = a_y[i]
        
        prev_m_z = m_z[i]
        
        prev_F = F[i]
        prev_D_x = D_x[i]
        prev_D_y = D_y[i]
        
        prev_alpha = alpha[i]
        prev_omega = omega[i]
        prev_theta = theta[i]
        
        prev_fuel = rem_fuel[i]
        
        prev_ballast_mass = ballast_mass[i]
        
        ## time
        t = np.append(t, prev_t + time_step)
        cur_t = prev_t + time_step
        
        ## Modified perpendicular area
        perp_A_x, perp_A_y, cent_x, cent_y = calc_geom(threeD_balloon, prev_theta)
        
        ## COG, COD, J (not rotated)
        COG_balloon_h = (payload_height + connector_height + balloon_height/2)
        COG_balloon_w = cent_x
        
        COG_cur_h = ((r_m[i] - balloon_mass)*COG_payload_h + balloon_mass*COG_balloon_h)/(r_m[i]) # calculates changing height COG
        COG_cur_w = ((r_m[i] - balloon_mass)*COG_payload_w + balloon_mass*COG_balloon_w)/(r_m[i]) # calculates changing COG
        
        J_payload_u = r_m[i]*(payload_height**2 + payload_width**2) # untransformed moment of inertia of payload
        trans_payload_J_d = np.sqrt(COG_cur_h**2 + COG_cur_w**2) - COG_payload # distance axis of rotation must be moved
        J_payload_t = J_payload_u + r_m[i]*trans_payload_J_d**2 # moving axis of rotation with parallel axis theorem
            
        trans_balloon_J_d = np.sqrt((COG_balloon_h - COG_cur_h)**2 + (COG_balloon_w - COG_cur_w)**2) # distance axis of rotation must be moved
        J_balloon_t = J_balloon_u + balloon_mass*trans_balloon_J_d**2 # moving axis of rotation with parallel axis theorem
            
        J_tot = J_payload_t + J_balloon_t    
        
        COD_balloon_h = COG_balloon_h # needs to be updated based on CFD
        COD_balloon_w = COG_balloon_w # needs to be updated based on CFD
        
        if prev_v_x != 0:
            re_num = rho_atmo*prev_v_x*dim_scale/dyn_visc
            C_f = .027/np.power(re_num, 1/7) ## Prandtl's 1/7 Power Law
        else:
            C_f = 0
        
        D_mag = np.sqrt(prev_D_x**2 + prev_D_y**2)
        
        res_freq = int(np.ceil(2*pi*np.sqrt(J_tot/(F_b*balloon_height))))
        max_alpha = max_theta/4*res_freq**2
        
        if cur_t < cutoff_time:
            reverse = 0
        else:
            reverse = 1
        
        if reverse:
            thrust = 0
            m_dot = 0
            curdtr = 0
            
        else:
            thrust = thrust_f
            m_dot = m_dot_f
            curdtr = abs(D_mag/thrust)
            
            
        if curdtr < dragthrustratio:
           if reverse:
               burn_condition = 0  
           else:
               burn_condition = 1
        else:
            if burn_index > min_burn_index:
                burn_condition = 0    
                burn_index = 0 
        
        if burn_condition:
            burn_index = burn_index + 1
            
            ## Force
            cur_F = thrust
            
            cur_fuel = prev_fuel - m_dot*time_step
            # Ballast
            cur_ballast_mass = prev_ballast_mass + m_dot*time_step
            
            cur_r_m = prev_r_m
            
        else:
            cur_F = 0
            cur_r_m = prev_r_m
            
            cur_fuel = prev_fuel
            
            mass_deficit = 0
            cur_ballast_mass = prev_ballast_mass
            
        perp_A_pay_x = payload_width/np.cos(prev_theta)*payload_depth
            
        pay_drag_x = -.5*(C_D_payload+C_f)*perp_A_pay_x*rho_atmo*prev_v_x**2
                
        ball_drag_x = -.5*(C_D_balloon+C_f)*perp_A_x*rho_atmo*prev_v_x**2
        ball_drag_y = -.5*(C_D_balloon+C_f)*perp_A_y*rho_atmo*prev_v_y**2
        
        cur_D_x = pay_drag_x + ball_drag_x
        cur_D_y =  ball_drag_y
        cur_D_mag = np.sqrt(cur_D_x**2 + cur_D_y**2)
        
        ## Linear Kinematics
        tot_force_x = cur_F*np.cos(prev_theta) + cur_D_x
        tot_force_y = cur_F*np.sin(prev_theta) + cur_D_y
        
        cur_a_x = tot_force_x/cur_r_m
        cur_a_y = tot_force_y/cur_r_m
        
        cur_v_x = prev_v_x+cur_a_x*time_step
        cur_v_y = prev_v_y+cur_a_y*time_step
        
        cur_d_x = prev_d_x+cur_v_x*time_step
        cur_d_y = prev_d_y+cur_v_y*time_step
        
        
        ## Rotational Kinematics
        # Payload Gravity Torque
        g_m_a_y_pay = COG_cur_h - COG_payload_h # moment arm for gravity on the payload y
        g_m_a_x_pay = COG_cur_w - COG_payload_w # moment arm for gravity on the payload x
        g_m_a_pay = np.sqrt(g_m_a_y_pay**2 + g_m_a_x_pay**2)
        g_m_pay = abs((cur_r_m - balloon_mass)*acc_g * np.sin(prev_theta) * g_m_a_pay)
        
        # Balloon Gravity Torque
        g_m_a_y_ball = COG_cur_h - COG_balloon_h # moment arm for gravity on the payload y
        g_m_a_x_ball = COG_cur_w - COG_balloon_w # moment arm for gravity on the payload x
        g_m_a_ball = np.sqrt(g_m_a_y_pay**2 + g_m_a_x_pay**2)
        g_m_ball = -abs((cur_r_m - balloon_mass)*acc_g * np.sin(prev_theta) * g_m_a_ball)
        
        g_m = g_m_pay + g_m_ball
        
        # Balloon Drag Torque
        d_m_a_y = COD_balloon_h - COG_cur_h # moment arm for drag on the balloon y
        d_m_a_x = COD_balloon_w - COG_cur_w # moment arm for drag on the balloon x
        d_m_a = np.sqrt(d_m_a_y**2 + d_m_a_x**2) # euclidean distance
        
        ball_D_mag = np.sqrt(ball_drag_x**2 + ball_drag_y**2) # magnitude of drag on balloon
        
        d_m = d_m_a*ball_D_mag*np.cos(prev_theta) - pay_drag_x*g_m_a_pay # sum all drag moments
        
        # Bouyancy force torque, balloon
        b_m_a_y = COG_balloon_h - COG_cur_h # moment arm for bouyancy force y 
        b_m_a_x = COG_balloon_w - COG_cur_w # moment arm for bouyancy force x
        b_m_a = np.sqrt(b_m_a_y**2 + b_m_a_x**2) # euclidean 
        
        b_m = b_m_a * F_b * np.sin(prev_theta) # total buoyancy moment
        
        t_m_a = moment_arm_thruster # thruster moment arm
        t_m = cur_F * (moment_arm_thruster) # thruster moment
        
        m_z_tot = d_m - b_m + t_m - g_m # total moment
        cur_alpha = m_z_tot / J_tot
        cur_omega = prev_omega + cur_alpha*time_step
        cur_theta = prev_theta + cur_omega*time_step
        
        ## all updates
        F = np.append(F, cur_F)
        r_m = np.append(r_m, cur_r_m)
        
        D_x = np.append(D_x, cur_D_x)
        D_y = np.append(D_y, cur_D_y)  
        
        a_x = np.append(a_x, cur_a_x)
        a_y = np.append(a_y, cur_a_y)
        
        v_x = np.append(v_x, cur_v_x)
        v_y = np.append(v_y, cur_v_y)
            
        d_x = np.append(d_x, cur_d_x)
        d_y = np.append(d_y, cur_d_y)
        
        m_z = np.append(m_z, m_z_tot)
        alpha = np.append(alpha, cur_alpha)
        omega = np.append(omega, cur_omega)
        theta = np.append(theta, cur_theta) 
        
        rem_fuel = np.append(rem_fuel, cur_fuel)
        
        ballast_mass = np.append(ballast_mass, cur_ballast_mass)
        
        i = i + 1
        if cur_fuel < 0:
            fail = 1
            print('Not Enough Fuel Mass')
        
        if i % 100 == 0:
            print('.', end= '')
            
        if i % 5000 == 0:
            print('\n')
            
        if i > ind_ignore:
            vel_checker = prev_v_x
        else:
            vel_checker = 10 # lets loop accelerate the rover
            
    acheived_disp = d_x[-1]
    
    all_data = np.zeros((len(t), 17))
    
    all_data[:, 0] = t
    all_data[:, 1] = F
    all_data[:, 2] = r_m
    all_data[:, 3] = D_x
    all_data[:, 4] = D_y
    all_data[:, 5] = a_x
    all_data[:, 6] = a_y
    all_data[:, 7] = v_x
    all_data[:, 8] = v_y
    all_data[:, 9] = d_x
    all_data[:, 10] = d_y
    all_data[:, 11] = m_z
    all_data[:, 12] = alpha
    all_data[:, 13] = omega
    all_data[:, 14] = theta
    all_data[:, 15] = rem_fuel
    all_data[:, 16] = ballast_mass
    
    headers = ['time', 'force', 'mass', 'drag_x', 'drag_y', 'acceleration_x', 'acceleration_y', 'velocity_x', 'velocity_y', 'displacement_x', 'displacement_y', 'moment_z', 'alpha', 'omega', 'theta', 'fuel_mass', 'ballast_mass']
    
    if test:
        return acheived_disp
    else:
        return acheived_disp, pd.DataFrame(all_data, columns=headers)

## Constants ##
g_0 = 9.8 # Earth gravity (m/s^2)
G = 6.67E-11 # Gravitational Constant

## Rover Constants ##
altitude = 300 # hovering altitude m
payload_mass = 155 # rover payload mass kg
structure_mass = 239.65 # balloon mass kg


###### Body Info #####
body_atmo_pressure = 146921.2 # Atmospheric Pressure of visited body (Pa)
body_mass = 1.3452E23 # mass of body (kg)
body_radius = 2574.7*1000 # radius of body (m)
acc_g = G*body_mass/body_radius**2 # acceleration due to gravity (m/s^2)

rho_atmo = 1.225 * 4.4 #  density of atmosphere of body kg/m^3
dyn_visc = 4.35

## forward engine
thrust_per_engine_f = 3.6 # thrust per engine (N)
num_engines_f = 8 # number of engines (should be greater that 2 for torqueing)
thrust_f = thrust_per_engine_f * num_engines_f # total thrust (N)

## backward engine
thrust_per_engine_r = 3 # thrust per engine (N)
num_engines_r = 1 # number of engines (should be greater that 2 for torqueing)
thrust_r = thrust_per_engine_r * num_engines_r # total thrust (N)

Isp = 57 # specific impulse of rocket propellant (s)
vexit = g_0 * Isp # exit velocity of propellant
m_dot_f = thrust_f/vexit # mass flow rate (kg/s) 
m_dot_r = thrust_r/vexit

fuel_mass = 95 # available fuel mass (kg)

## Balloon Constants
volume_req = 356.76 # volume required to lift the mass of itself and the payload (m^3)

balloon_mass = structure_mass # mass of balloon (kg)

total_rover_mass = payload_mass + balloon_mass # total initial mass of rover (kg)

F_b = total_rover_mass * acc_g

# Balloon Shape  - https://www.geogebra.org/m/ydGnFQ2c

x = symbols('x') # sympy symbol

filename = 'C:/Users/vargh/Desktop/Scripts/SHREC/goodyear_airfoil.csv' # generated dataset from link above
gy_geom = pd.read_csv(filename)
gy_x = np.array(gy_geom.iloc[:, 0])

# normalize to scale x to 1 and y accordingly
norm_fac = max(gy_x) # normalizing factor
gy_x = gy_x/norm_fac # normalize x
gy_y = np.array(gy_geom.iloc[:, 1])/norm_fac # normalize y

fit = np.polyfit(gy_x, gy_y, deg=4) # get polynomial fit
fit = np.flip(fit) # flip coefficients

geom_model = 0
for i in range(len(fit)):
    geom_model = geom_model + fit[i]*x**i # build symbolic model
    

unit_vol = float(pi*integrate(geom_model**2, (x, 0, 1)).evalf()) # find volume of unit base
vol_scale = volume_req/unit_vol # find volume scale factor

dim_scale = np.cbrt(vol_scale) # find dimension scale factor

sized_geom_model = dim_scale*geom_model.subs(x, 1/dim_scale*x) # create an equivalent symbolic model
where_max = float(solve(Eq(diff(sized_geom_model), 0))[0]) # find the max x
value_max = float(sized_geom_model.subs(x, where_max)) # find y at max x

# Create a 3D Model
sampled_x = np.linspace(0, dim_scale)  # linearly sampled x values
sampled_z = np.zeros(len(sampled_x)) # initialize z values
for i in range(len(sampled_x)):
    sampled_z[i] = sized_geom_model.subs(x, sampled_x[i]) # evaluate z values
    
circle_chunks = 30 # splits 2*pi radians into this many chunks
radians = np.linspace(0, float(2*pi), circle_chunks) # linearly samples radians
threeD_balloon = np.zeros((len(sampled_x)*circle_chunks, 3)) # initialize 3d matrix

for i in range(circle_chunks):
    cur_angle = radians[i] # current angle
    
    threeD_balloon[i*len(sampled_x):(i+1)*len(sampled_x), 0] = sampled_x # transformed x
    threeD_balloon[i*len(sampled_x):(i+1)*len(sampled_x), 1] = sampled_z*np.cos(cur_angle) # transformed y
    threeD_balloon[i*len(sampled_x):(i+1)*len(sampled_x), 2] = sampled_z*np.sin(cur_angle) # transformed z

## Aerodynamics
perp_A_x = float(pi*value_max**2) # terminal velocity along x axis
C_D_balloon = .19 # hardcoded, need to find a value
C_D_payload = 2.1

terminal_vel = np.sqrt(2*thrust_f/(C_D_balloon*rho_atmo*perp_A_x)) # maximum terminal velocity (m/s) (can decrease if perp area becomes larger as a result of theta)

## Sanity Check Calculations
max_burn_time = fuel_mass / m_dot_f # maximum amount of burn time

## Initial COG, J
payload_height = .75 # height of payload (m)
payload_width = 2 # width of payload (m)
payload_depth = .75 # depth of payload (m)

balloon_height = 2*value_max # height of balloon (m)
balloon_width = dim_scale # width of balloon (m)

connector_height = 0 # height of connector of balloon to payload (m)

COG_payload_h = payload_height/2 # COG in h for payload (m)
COG_payload_w = payload_width/2 # COG in w for payload (m)

COG_payload = np.sqrt(COG_payload_h**2+COG_payload_w **2) # euclidean distance

J_balloon_u = 1/5*balloon_mass*(balloon_height**2 + balloon_width**2) # moment of inertia of balloon about center, approximated as ellipsoid

moment_arm_thruster = .91 # moment arm of thruster (m)

## Burn Optimization Parameters
dragthrustratio = .01 # ratio of thrust to drag
target_range = 328.57 # target range (m)
percent_error = .01
min_burn_time = 2 # (s) shortest duration of time an engine can burn

time_step = .2 # simulation time step size

d_tol = target_range*percent_error # tolerance for displacement (m)
v_tol = terminal_vel*percent_error
theta_tol = .001 # tolerance for theta (rad)

## Calculation ##

max_theta = np.radians(60) #maximum theta
consider_bouyancy_drift = 1

min_burn_index = int(min_burn_time/time_step)
print('Calculating burn profile with no drag stops')
init_data = init_calc(threeD_balloon, payload_height, payload_width, payload_depth, connector_height, balloon_height, balloon_mass, COG_payload_h, COG_payload_w, rho_atmo, dim_scale, dyn_visc, F_b, thrust_f, thrust_r, m_dot_f, m_dot_r, acc_g, consider_bouyancy_drift, time_step, target_range, d_tol, dragthrustratio, min_burn_index, moment_arm_thruster)
print('\nInitial calculations complete\n')

maneuver_time = init_data['time'].iloc[-1]
max_vel = max(init_data['velocity_x'])

vel_elbow = .1 * max_vel
ind_ignore = np.where(init_data['velocity_x'] > vel_elbow)[0][0]
ind_at_end = int(maneuver_time/time_step) + 1

f_b_f = np.linspace(ind_ignore/ind_at_end, .99, 3)
d_list = np.zeros((3))

print('Calculating cutoff time')
for i in range(len(f_b_f)):
    print('\n  - Iteration %d of %d'%(i+1, len(f_b_f)))
    test = 1
    forward_burn_frac = f_b_f[i]
    tmp = drag_stop_calc(1, ind_ignore, maneuver_time, vel_elbow, forward_burn_frac, ind_at_end, threeD_balloon, payload_height, payload_width, payload_depth, connector_height, balloon_height, balloon_mass, COG_payload_h, COG_payload_w, rho_atmo, dim_scale, dyn_visc, F_b, thrust_f, thrust_r, m_dot_f, m_dot_r, acc_g, consider_bouyancy_drift, time_step, target_range, d_tol, dragthrustratio, min_burn_index, moment_arm_thruster)
    d_list[i] = tmp
    
interpolator = interp1d(d_list, f_b_f)
correct_f_b_f = interpolator(target_range)

print('\nCalculating Correct Data')
tmp, correct_data =  drag_stop_calc(0, ind_ignore, maneuver_time, vel_elbow, correct_f_b_f, ind_at_end, threeD_balloon, payload_height, payload_width, payload_depth, connector_height, balloon_height, balloon_mass, COG_payload_h, COG_payload_w, rho_atmo, dim_scale, dyn_visc, F_b, thrust_f, thrust_r, m_dot_f, m_dot_r, acc_g, consider_bouyancy_drift, time_step, target_range, d_tol, dragthrustratio, min_burn_index, moment_arm_thruster)
