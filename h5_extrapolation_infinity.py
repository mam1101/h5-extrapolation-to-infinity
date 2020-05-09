"""
Title: h5 Extrapolation
Author: Max Miller
Last Modfied: 3/25/2020
Description:
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.integrate as integrate
import pdb
import os
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
#from sympy.calculus.singularities import singularities
#from sympy import Symbol
from pandas import DataFrame
import pandas as pd
from collections import OrderedDict
from sys import argv
import argparse

## State variables
NO_TITLE = True
BUILD_FULL_PLOTS = True
NO_T_CORR = False
RERUN = True
PLOT_FILE_TYPE = 'pdf'

file_additional_str = ""
if NO_T_CORR:
    file_additional_str = "_no_t_corr"

def r_star(r, M_ADM):
    return r + 2*M_ADM*np.log((r/(2*M_ADM)) - 1)


def t_corr(g_TT, R_areal, M_ADM):
    return np.sqrt((-1/g_TT)/(1-(2*M_ADM/R_areal)))


def A(h_p, h_c):
    return np.sqrt(h_p**2 + h_c**2)


# k is solved for (k should be an integer which makes phi continuous (b/c arctan is NOT continuous))
def phi(h_p, h_c, k):
    return np.arctan2(h_c, h_p) + 2*np.pi*k

# Integrate over T' and plug in known values for each point in the integration. Compare this t_corr to the original t
# See if this dependence matters over time.

# The h5 file being read
f_name = '/pylon5/ph4s83p/ffoucart/MaxTest/WaveL2/rh_FiniteRadii_CodeUnits.h5'
f = h5py.File(f_name, 'r')

# Defines the time step for calculations and plotting involving t_ret
t_ret_interval = 100
t_ret_range = range(0, 13701, t_ret_interval)

caching_directory = "cached_calculations"

# Gets all the coordinate radii for the given h5 directory
radii = f.keys()

A_int_set = [[0 for x in range(len(radii))] for y in t_ret_range]
phi_int_set = [[0 for x in range(len(radii))] for y in t_ret_range]
radii_list = [[0 for x in range(len(radii))] for y in t_ret_range]

M_ADM = 5.561165775

# Ensures the radii are in the correct order
radii.sort()

# Check for any cached files
if not RERUN and (not NO_T_CORR \
    and os.path.exists("{}/phi_set_{}.csv".format(caching_directory, t_ret_interval)) \
    and os.path.exists("{}/a_set_{}.csv".format(caching_directory, t_ret_interval)) \
    and os.path.exists("{}/radii_set_{}.csv".format(caching_directory, t_ret_interval))):

    print("Cached files found...")

    phi_int_set = pd.read_csv("{}/phi_set_{}.csv".format(caching_directory, t_ret_interval)).values
    A_int_set = pd.read_csv("{}/a_set_{}.csv".format(caching_directory, t_ret_interval)).values
    radii_list = pd.read_csv("{}/radii_set_{}.csv".format(caching_directory, t_ret_interval)).values
    print("Cached files loaded")
elif not RERUN and (NO_T_CORR \
    and os.path.exists("{}/phi_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str)) \
    and os.path.exists("{}/a_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str)) \
    and os.path.exists("{}/radii_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str))):

    print("Cached files found (no t_corr)...")

    phi_int_set = pd.read_csv("{}/phi_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str)).values
    A_int_set = pd.read_csv("{}/a_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str)).values
    radii_list = pd.read_csv("{}/radii_set_{}{}.csv".format(caching_directory, t_ret_interval, file_additional_str)).values
    print("Cached files loaded (no t_corr)")
else:
    try:
        os.makedirs(caching_directory)
        print("Directory " , caching_directory ,  " Created ")
    except Exception:
        pass
    # Loop tracking index
    index = 0
    # Step through all coordinate radii
    for radius in radii:
        print "{}: Radius {}...".format(index, radius)
        waveform = 'Y_l2_m-2.dat'
        current_wave = f[radius][waveform]

        avg_lapse = f[radius]['AverageLapse.dat'] # T' dependent
        R_areal_dat = f[radius]['ArealRadius.dat']  # T' dependent
        coor_radius = f[radius]['CoordRadius.dat']

        g_TT_dat = avg_lapse

        data_set_len = len(R_areal_dat[0:, 1])

        x = current_wave[0:, 0]
        y = current_wave[0:, 1]

        intF = np.zeros(data_set_len)
        t_ret = np.zeros(data_set_len)

        A_set = np.zeros(data_set_len)
        phi_set = np.zeros(data_set_len)

        A_set[0] = A(current_wave[0, 1], current_wave[0, 2])
        phi_set[0]  = phi(current_wave[0, 1], current_wave[0, 2], 1)

        k = 0

        for i in range(0, data_set_len-1):
            c_g_TT = -1/(g_TT_dat[i+1, 1])**2
            c_R_areal = R_areal_dat[i+1, 1]

            p_g_TT = -1/(g_TT_dat[i, 1])**2
            p_R_areal = R_areal_dat[i, 1]

            T = R_areal_dat[i, 0]

            c_r_star = r_star(c_R_areal, M_ADM)
            if not NO_T_CORR:
                # Trapeze rule integral calculation
                del_t = R_areal_dat[i + 1, 0] - T
                intF[i+1] = intF[i] + ((t_corr(c_g_TT, c_R_areal, M_ADM) + t_corr(p_g_TT, p_R_areal, M_ADM))*del_t)/2
                t_ret[i+1] = intF[i+1] - c_r_star
            else:
                t_ret[i+1] = current_wave[i+1, 0] - c_r_star

            A_check = A(current_wave[i+1, 1], current_wave[i+1, 2])
            phi_check = phi(current_wave[i+1, 1], current_wave[i+1, 2], k)

            # Solving for k in phi calculation
            # if previous phi is +/- pi away, then add or remove 2pi until new phi is within +/- pi from old phi
            if np.abs(phi_check - phi_set[i]) > np.pi:
                if phi_check < phi_set[i] + np.pi:
                    while np.abs(phi_check - phi_set[i]) > np.pi:
                        k += 1
                        phi_check = phi(current_wave[i+1, 1], current_wave[i+1, 2], k)
                else:
                    while np.abs(phi_check - phi_set[i]) > np.pi:
                        k -= 1
                        phi_check = phi(current_wave[i+1, 1], current_wave[i+1, 2], k)

            A_set[i+1] = A_check
            phi_set[i+1]  = phi_check

            h_p = current_wave[i+1, 1]
            h_x = current_wave[i+1, 2]

            check_h_x = A_check * np.sin(phi_check)
            check_h_p = A_check * np.cos(phi_check)

            if np.abs(h_p/check_h_p) < 0.99 or np.abs(h_x/check_h_x) < 0.99:
                raise Exception("AmplitudeError: Parameter not within check threshold.")

        ##########################################
        #phi of constant t_r across all radii (interpolate the phi values)
        ##########################################

        interp_phi = interp1d(t_ret, phi_set)
        interp_A = interp1d(t_ret, A_set)
        interp_R_areal = interp1d(t_ret, R_areal_dat[0:, 1])

        for ret in t_ret_range:
            phi_int_set[ret/t_ret_interval][index] = (interp_phi(ret))
            A_int_set[ret/t_ret_interval][index] = (interp_A(ret))
            if interp_R_areal(ret) == 0:
                radii_list[ret/t_ret_interval][index] = 0
            else:
                radii_list[ret/t_ret_interval][index] = (1/interp_R_areal(ret))*100

        phase_plot_dir = "phase_plots/{}".format(PLOT_FILE_TYPE)
        try:
            os.makedirs(phase_plot_dir)
            print("Directory " , phase_plot_dir ,  " Created ")
        except Exception:
            pass

        fig, axs = plt.subplots(2, 1, figsize=(15,15))
        axs[0].set_xlabel('t')
        axs[1].set_xlabel('t')
        axs[0].set_ylabel('$h_+$')
        axs[1].set_ylabel('$h_X$')

        axs[0].plot(current_wave[0:, 0], current_wave[0:, 1], label='Waveform simulated')
        axs[1].plot(current_wave[0:, 0], current_wave[0:, 2], label='Waveform simulated')
        # axs[0].plot(t_ret, current_wave[0:, 1], '--', color='red', label='Waveform with $t_{ret}$')
        # axs[1].plot(t_ret, current_wave[0:, 2], '--', color='red', label='Waveform with $t_{ret}$')
        axs[0].legend()
        axs[1].legend()
        plt.savefig('{0}/phase_plot__{1}.{2}'.format(phase_plot_dir, radius, PLOT_FILE_TYPE))
        plt.close()

        index += 1

    # Creates CSV for caching purposes
    DataFrame(phi_int_set).to_csv('{}/phi_set_{}{}.csv'.format(caching_directory, t_ret_interval, file_additional_str), index=False)
    DataFrame(A_int_set).to_csv('{}/a_set_{}{}.csv'.format(caching_directory, t_ret_interval, file_additional_str), index=False)
    DataFrame(radii_list).to_csv('{}/radii_set_{}{}.csv'.format(caching_directory, t_ret_interval, file_additional_str), index=False)

"""
Start of fitting functions
"""
def extrap_fit_2_fit(x, A_0, A_1, A_2):
    return A_0 + A_1*(x) + A_2*(np.power(x,2))


def extrap_fit_3_fit(x, A_0, A_1, A_2, A_3):
    return A_0 + A_1*(x) + A_2*(np.power(x,2)) + A_3*(np.power(x,3))


def extrap_fit_4_fit(x, A_0, A_1, A_2, A_3, A_4):
    return A_0 + A_1*(x) + A_2*(np.power(x,2)) + A_3*(np.power(x,3)) + A_4*(np.power(x,4))


def extrap_fit_5_fit(x, A_0, A_1, A_2, A_3, A_4, A_5):
    return A_0 + A_1*(x) + A_2*(np.power(x,2)) + A_3*(np.power(x,3)) + A_4*(np.power(x,4)) + A_5*(np.power(x,5))

extrap_functions = [extrap_fit_2_fit, extrap_fit_3_fit, extrap_fit_4_fit, extrap_fit_5_fit]
"""
End of fitting functions
"""

"""
Start of extrapolation
"""
## Allows for changing which radii the extrapolation will end with (i.e. [0, 2] means run once with all radii, then run once without the last two radii)
starting_radii = [0]

## Array used to compare extrapolation methods
order_list = [{"phi": [[] for x in range(0, len(extrap_functions))], "A": [[] for x in range(0, len(extrap_functions))]} for i in range(0, len(starting_radii))]
## Values of t_ret to be used in graphing
t_ret_vals = [x for x in t_ret_range]

for i in range(0, len(radii_list)): # loop over each t_ret (i*t_ret_interval == t_ret)
    x_vals_list = [] # The current x_vals used for graphing
    radii_list[i] = radii_list[i] / max(radii_list[i])
    print("Proccessing {}...".format(i))
    if BUILD_FULL_PLOTS:
        fig, axs = plt.subplots(1, 2, figsize=(15,5))
    for y in range(0, len(extrap_functions)):   # loop over n order where y=n
        extrap_vals = []
        # Allows for removal of starting radii in extrapolation calculations
        q = 0
        if not BUILD_FULL_PLOTS:
            fig, axs = plt.subplots(1, 2, figsize=(15,5))
        for r in starting_radii:
            file_additional_str = "_r{}".format(r)
            if not BUILD_FULL_PLOTS:
                dirName = "interp_wave_plots_{0}_{1}/{2}".format(y+2, t_ret_interval, PLOT_FILE_TYPE)
            else:
                dirName = "interp_wave_plots_full/{0}".format(PLOT_FILE_TYPE)

            try:
                os.makedirs(dirName)
                print("Directory " , dirName ,  " Created ")
            except Exception:
                pass
                # print("Directory " , dirName ,  " already exists")

            diff_arr__p = []
            diff_arr__A = []

            ## Performs the extraplation
            popt_p, pcov_p = curve_fit(extrap_functions[y], radii_list[i][:len(radii_list[i]) - r], phi_int_set[i][:len(radii_list[i]) - r])
            popt_a, pcov_a = curve_fit(extrap_functions[y], radii_list[i][:len(radii_list[i]) - r], A_int_set[i][:len(radii_list[i]) - r])

            phi_graph_vals = extrap_functions[y](
                0,
                *popt_p
                )
            A_graph_vals = extrap_functions[y](
                0,
                *popt_a
                )

            x_vals = np.arange(0, max(radii_list[i]), 0.00001)

            extrap_val_a = extrap_functions[y](
                x_vals,
                *popt_a
                )

            extrap_val_phi = extrap_functions[y](
                x_vals,
                *popt_p
                )

            axs[0].plot(x_vals,
                extrap_val_phi,
                label="${\phi}_{" + str(y+2) + "}$"
                )
            axs[0].plot(radii_list[i][:len(radii_list[i]) - r], phi_int_set[i][:len(radii_list[i]) - r], 'b.')
            axs[0].set_xlabel('1/r (Normalized)')
            axs[0].set_ylabel(r'${\phi}(h_+, h_x)$')
            if not NO_TITLE:
                axs[0].set_title("Phi vs. Raddii ({}) at t_ret={} - FIT ORDER: {}".format(len(radii_list[i])-r, t_ret_interval*i, y+2))
            axs[1].plot(radii_list[i][:len(radii_list[i]) - r], A_int_set[i][:len(radii_list[i]) - r], 'b.')
            axs[1].plot(x_vals,
                extrap_val_a,
                label="$A_{" + str(y+2) + "}$"
                )
            axs[1].set_xlabel('1/r (Normalized)')
            axs[1].set_ylabel(r'$A(h_+, h_x)$')
            if not NO_TITLE:
                axs[1].set_title("A vs. Raddii ({}) at t_ret={} - FIT ORDER: {}".format(len(radii_list[i])-r, t_ret_interval*i, y+2))
            axs[0].legend()
            axs[1].legend()
            ## Adds data to the array used for extrapolation order comparison graphs (see end)
            order_list[q]["phi"][y].append(phi_graph_vals)
            order_list[q]["A"][y].append(A_graph_vals)
            q+=1
        if not BUILD_FULL_PLOTS:
            plot_name = "phi_A_comparison"
            plt.savefig('{0}/{1}_{2}__{3}{4}.{5}'.format(dirName, plot_name, t_ret_interval*i, y+2, file_additional_str, PLOT_FILE_TYPE))
            plt.close()
    if BUILD_FULL_PLOTS:
        plot_name = "phi_A_comparison_full"
        plt.savefig('{0}/{1}_{2}{3}.{4}'.format(dirName, plot_name, t_ret_interval*i, file_additional_str, PLOT_FILE_TYPE))
        plt.close()


## Creates a directory for the extrapolation comparison plots if none exists
try:
    os.makedirs("extrap_comparisons/{}".format(PLOT_FILE_TYPE))
    print("Directory " , "extrap_comparisons",  " Created ")
except Exception:
    pass

order_df = {}
last_time = t_ret_vals.index(13000)
## Creates the plots for comparing the different order extrapolation methods (n and n-1)
for q in range(0, len(starting_radii)):
    fig, axs = plt.subplots(1, 2, figsize=(15,5))
    for n in range(1, len(extrap_functions)):
        axs[0].plot(
        t_ret_vals[:last_time],
        np.subtract(np.array(order_list[q]["phi"][n]), np.array(order_list[q]["phi"][n-1]))[:last_time],
        label="${\phi}_{" + str(n+2) + "}-{\phi}_{" + str(n+1) + "}$")
        # axs[0].set_title("Phi Extrap Values Comparison Total Radii={}".format(len(radii_list[-1]) - starting_radii[q]))
        axs[0].set_xlabel(r"$t_{ret}$")
        axs[0].set_ylabel(r"${\phi}_{n}-{\phi}_{n-1}$")
        axs[0].legend()
        axs[1].plot(
        t_ret_vals[:last_time],
        np.subtract(np.array(order_list[q]["A"][n]), np.array(order_list[q]["A"][n-1]))[:last_time],
        label="$A_{" + str(n+2) + "}-A_{" + str(n+1) + "}$")
        # axs[1].set_title("A Extrap Values Comparison Total Radii={}".format(len(radii_list[-1]) - starting_radii[q]))
        axs[1].set_xlabel(r"$t_{ret}$")
        axs[1].set_ylabel(r"$A_{n}-A_{n-1}$")
        axs[1].legend()
        order_df["{}".format(n)] = []
    plt.savefig('{0}/{1}.{2}'.format("extrap_comparisons", "extrap_comp_main_{}_{}{}".format(t_ret_interval, len(radii_list[-1]) - starting_radii[q], file_additional_str), PLOT_FILE_TYPE))
    plt.close()
