#Set up the correct directory to access data and utilities

import os
import sys
current_folder = os.path.dirname(os.path.abspath("__file__")) # Gets the location of the present script
parent_folder = os.path.dirname(current_folder) # Gets the path for the origin of the repository, as it is locally stored
sys.path.append(os.path.join(parent_folder, 'utilities'))
#utils_folder = os.path.join(parent_folder, 'utilities') # Creates a view to the utilites the folder
#print(current_folder)
#os.chdir(utils_folder) # Makes it the active folder
from metro_utils import * # Imports everything from metro_utils
from rf_utils import * # Imports everything from rf_utils
#from LabberReader import * # Uncomment if you have access to the Labber API
os.chdir(current_folder) # Makes the folder of the script the active one

import xarray as xr 
import matplotlib.pyplot as plt
import numpy as np

#takes as input argument the profile for the computation of fft
#finds the optimal window to perform the moving average algorithm
def fft_for_window_np(frequency_array: np.ndarray,
                      profile_array: np.ndarray,
                      pump_frequency: float,
                      cut_off: float=11.5e9,
                      step: float=16e6,
                      perc_off: float=0.1,
                      alpha: float=0.06):
    
    #Move 200 MHz to the left and find the max of the left region
    #For pump frequencies near 4 GHz we have a ValueError: in this case find the max for f > pump_frequency
    try:
        cut = pump_frequency-200e6
        massimo_sx = np.argmax(profile_array[frequency_array <= cut]) #index of the max of the region
        max_freq = frequency_array[massimo_sx] #frequency of the max of the profile
        threshold = max(pump_frequency+200e6, 5e9)

        #if the interval length would be less than 1 GHz, move to the right of the pump frequency
        if max_freq <= 5e9:
            massimo_sx_uncut = profile_array[frequency_array >= threshold]
            freq_thr = frequency_array[frequency_array >= threshold]
            massimo_sx = np.argmax(massimo_sx_uncut[freq_thr <= cut_off])
            max_freq = frequency_array[massimo_sx+(len(frequency_array)-len(massimo_sx_uncut))]

            #there could be spikes in the profile array at frequencies which are integer multiples of the pump frequency
            #avoid that the maximum is in these points
            if max_freq == frequency_array[frequency_array == 2*pump_frequency]:
                massimo_sx = np.argmax(massimo_sx_uncut[freq_thr <= 2*pump_frequency-200e6])
                max_freq = frequency_array[massimo_sx+(len(frequency_array)-len(massimo_sx_uncut))]
    except ValueError:
        massimo_dx_uncut = profile_array[frequency_array >= threshold]
        freq_thr = frequency_array[frequency_array >= threshold]
        massimo_dx = np.argmax(massimo_dx_uncut[freq_thr <= cut_off])
        max_freq = frequency_array[massimo_dx+(len(frequency_array)-len(massimo_dx_uncut))]
        
        if max_freq == frequency_array[frequency_array == 2*pump_frequency]:
            massimo_dx = np.argmax(massimo_dx_uncut[freq_thr <= 2*pump_frequency-200e6])
            max_freq = frequency_array[massimo_dx+(len(frequency_array)-len(massimo_dx_uncut))]

    #Zoom in a 1 GHz-wide region where we will perform the fft
    lim_sup = profile_array[frequency_array <= max_freq]
    lim_sup_freq = frequency_array[frequency_array <= max_freq]
    lim1 = lim_sup[lim_sup_freq >= max_freq-1e9]
    lim1_freq = lim_sup_freq[lim_sup_freq >= max_freq-1e9]
    if max_freq >= 5e9:
        max1 = np.mean(lim1)
    else:
        max1 = float("-inf")
    
    lim_inf = profile_array[frequency_array >= max_freq]
    lim_inf_freq = frequency_array[frequency_array >= max_freq]
    lim2 = lim_inf[lim_inf_freq <= max_freq+1e9]
    lim2_freq = lim_inf_freq[lim_inf_freq <= max_freq+1e9]
    max2 = np.mean(lim2)

    #compare max1 and max2 to find the region where the profile is more flat

    if abs(max1) > abs(max2):
        lim = lim1
        zoom = lim1_freq
    else:
        lim = lim2
        zoom = lim2_freq

    a, b = zoom[0], zoom[-1]
    print(f"Zoom start: {a} Hz\nZoom end: {b} Hz")

    #compute fft
    fft = np.fft.fft(lim) #x axis
    magnitude = np.abs(fft) #y axis

    #x axis range
    frequencies = np.linspace(a, b+8e6, len(lim))
    time = frequencies
    dt = np.mean(np.diff(time))
    fft_frequencies = np.fft.fftfreq(len(lim), d=dt)

    #find the max of the fft and its index 
    massimo_trasf = np.max(magnitude[int(perc_off*len(fft_frequencies)//2):len(fft_frequencies)//2])
    i0 = np.argmax(magnitude[int(perc_off*len(fft_frequencies)//2):len(fft_frequencies)//2])

    #find the period
    freq_trasf = 1/fft_frequencies[int(perc_off*len(fft_frequencies)//2):len(fft_frequencies)//2][i0]
    print(f"Max(fft) = {massimo_trasf}\nTime = {1/freq_trasf} s\nFreq = {freq_trasf} Hz")

    #the optimal window is proportional to the product massimo_trasf*freq_trasf
    if int(alpha*massimo_trasf*freq_trasf/step)%2 == 1 and int(alpha*massimo_trasf*freq_trasf/step) > 1:
        window_opt = int(alpha*massimo_trasf*freq_trasf/step)
    elif int(alpha*massimo_trasf*freq_trasf/step) == 1:
        window_opt = int(alpha*massimo_trasf*freq_trasf/step)+2
    else:
        window_opt = int(alpha*massimo_trasf*freq_trasf/step)+1
    
    #correction for low values of massimo_trasf and freq_trasf
    if window_opt < 5:
        window_opt += 2

    print(f"Optimal window width (points): {window_opt}")
    fig, ax = plt.subplots(2, 1, figsize=[10, 5], dpi=200)

    #profile plot
    ax[0].plot(time, lim, color="red", label="Profile")
    ax[0].set_xlabel("Frequency (Hz)") 
    ax[0].set_ylabel("|S21| / dB")
    ax[0].legend()
    ax[0].grid()

    #fft plot
    ax[1].plot(fft_frequencies[int(perc_off*len(fft_frequencies)//2):len(fft_frequencies)//2], magnitude[int(perc_off*len(fft_frequencies)//2):len(fft_frequencies)//2], label="FFT")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Magnitude")
    ax[1].legend()
    ax[1].grid()

    fig.tight_layout()
    plt.show()

    return window_opt, massimo_trasf, freq_trasf


#obtain the smoothed profile from the original profile using moving average
def padded_moving_average_np(profile_array: np.ndarray,
                              window: int=11):
    #correction added to include in the smoothing process the first and last (window//2) points
    padded_data = np.pad(profile_array, (window//2, window//2), mode='reflect')
    smuss = profile_array.copy()
    smuss = np.convolve(padded_data, np.ones(window)/window, mode="valid")
    return smuss

#finds the gain values for frequencies < pump frequency and frequencies > pump frequency
def find_gain_np(frequency_array: np.ndarray,
              smoothed_profile: np.ndarray,
              pump_frequency: float, 
              step: float=16e6, 
              n_step: int=10, 
              cut_off: float=1.15e10):
    
    #for values of pump_frequency near 4 GHz we have a ValueError
    try:
        #max in the left region
        G_LRB = np.max(smoothed_profile[frequency_array < pump_frequency-n_step*step])

        #max in the right region
        G_URB_notcut = smoothed_profile[frequency_array > pump_frequency+n_step*step]
        #there might be a max for very high frequencies
        G_URB_freq = frequency_array[frequency_array > pump_frequency+n_step*step]
        G_URB = np.max(G_URB_notcut[G_URB_freq <= cut_off])
        G = max(G_LRB, G_URB)
    except ValueError:
        G_LRB = np.max(smoothed_profile[frequency_array <= pump_frequency])
        G_URB_notcut = smoothed_profile[frequency_array > pump_frequency+n_step*step]
        G_URB_freq = frequency_array[frequency_array > pump_frequency+n_step*step]
        G_URB = np.max(G_URB_notcut[G_URB_freq <= cut_off])
        G = max(G_LRB, G_URB)

    #Actual gain
    min_LRB = G_LRB-3
    min_URB = G_URB-3

    #consider only the case where there is actual amplification in progress, there is where the profile max is above 3 dB
    if min_LRB <= 0:
        min_LRB = 0
    if min_URB <= 0:
        min_URB = 0

    return min_LRB, min_URB


#once obtained min_LRB and min_URB, find the two readout bands
def find_bandwidth_np(frequency_array: np.ndarray,  
                      smoothed_profile: np.ndarray,
                      pump_frequency: float, 
                      step: float=16e6, 
                      n_step: int=10, 
                      cut_off: float=11.5e9):
    
    min_LRB, min_URB = find_gain_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
    
    #set the legth of the readout bands to 0 if there is no amplification
    if min_LRB == 0 and min_URB == 0:
        intersection_points_LRB = [[0, 0], [0, 0]]
        intersection_points_URB = [[0, 0], [0, 0]]

    #cases if only one band is relevant 
    elif min_LRB == 0:
        intersection_points_LRB = [[0, 0], [0, 0]]
        G_URB_index = np.where(smoothed_profile == min_URB+3)[0]

        #find the intersection points between the smoothed profile and the function y = min_URB
        zeros_URB = smoothed_profile - min_URB
        sign_changes_URB = np.where(np.diff(np.sign(zeros_URB)))[0]

        left_URB = None
        right_URB = None

        #take only the nearest intersection point to the max
        #to the left
        for i in reversed(sign_changes_URB):
            if i < G_URB_index:
                left_URB = i
                break

        #to the right
        for i in sign_changes_URB:
            if i > G_URB_index:
                right_URB = i
                break

        #Coordinates
        intersection_points_URB = []
        if left_URB is not None:
            intersection_points_URB.append((frequency_array[left_URB+1], smoothed_profile[left_URB+1]))
        if right_URB is not None:
            intersection_points_URB.append((frequency_array[right_URB], smoothed_profile[right_URB]))
        
        #Add the first point of the array if there is only one intersection point
        if len(intersection_points_URB) == 1:
            intersection_points_URB.insert(0, (4e9, smoothed_profile[frequency_array == 4e9]))

    elif min_URB == 0:
        intersection_points_URB = [[0, 0], [0, 0]]
        G_LRB_index = np.where(smoothed_profile == min_LRB+3)[0]

        #find the intersection points between the smoothed profile and the function y = min_LRB
        zeros_LRB = smoothed_profile - min_LRB
        sign_changes_LRB = np.where(np.diff(np.sign(zeros_LRB)))[0]

        left_LRB = None
        right_LRB = None

        for i in reversed(sign_changes_LRB):
            if i < G_LRB_index:
                left_LRB = i
                break

        for i in sign_changes_LRB:
            if i > G_LRB_index:
                right_LRB = i
                break

        #Coordinates
        intersection_points_LRB = []
        if left_LRB is not None:
            intersection_points_LRB.append((frequency_array[left_LRB+1], smoothed_profile[left_LRB+1]))
        if right_LRB is not None:
            intersection_points_LRB.append((frequency_array[right_LRB], smoothed_profile[right_LRB]))

        if len(intersection_points_LRB) == 1:
            intersection_points_LRB.insert(0, (4e9, smoothed_profile[frequency_array == 4e9][0]))

    else:
        G_LRB_index = np.where(smoothed_profile == min_LRB+3)[0]
        G_URB_index = np.where(smoothed_profile == min_URB+3)[0]
        zeros_LRB = smoothed_profile - min_LRB
        sign_changes_LRB = np.where(np.diff(np.sign(zeros_LRB)))[0]

        zeros_URB = smoothed_profile - min_URB
        sign_changes_URB = np.where(np.diff(np.sign(zeros_URB)))[0]
        #the smooth profile value of the first index is always < min_LRB/min_URB so take index in position 1

        left_LRB = None
        right_LRB = None
        left_URB = None
        right_URB = None

        for i in reversed(sign_changes_LRB):
            if i < G_LRB_index:
                left_LRB = i
                break

        for i in sign_changes_LRB:
            if i > G_LRB_index:
                right_LRB = i
                break

        # Coordinates
        intersection_points_LRB = []
        if left_LRB is not None:
            intersection_points_LRB.append((frequency_array[left_LRB+1], smoothed_profile[left_LRB+1]))
        if right_LRB is not None:
            intersection_points_LRB.append((frequency_array[right_LRB], smoothed_profile[right_LRB]))

        for i in reversed(sign_changes_URB):
            if i < G_URB_index:
                left_URB = i
                break

        for i in sign_changes_URB:
            if i > G_URB_index:
                right_URB = i
                break

        # Coordinates
        intersection_points_URB = []
        if left_URB is not None:
            intersection_points_URB.append((frequency_array[left_URB+1], smoothed_profile[left_URB+1]))
        if right_URB is not None:
            intersection_points_URB.append((frequency_array[right_URB], smoothed_profile[right_URB]))

        if len(intersection_points_LRB) == 1:
            intersection_points_LRB.insert(0, (4e9, smoothed_profile[frequency_array == 4e9][0]))
        if len(intersection_points_URB) == 1:
            intersection_points_URB.insert(0, (4e9, smoothed_profile[frequency_array == 4e9][0]))
    
        #there are always two readout bands (left and right)
        #but it may happen that the intersection points for the two regions are the same
        #in this case we consider the LEFT readout band as the actual band and put the values for the right readout band to zero    
        one_band = False
        if intersection_points_LRB[0][0] == intersection_points_URB[0][0] and intersection_points_LRB[-1][0] == intersection_points_URB[-1][0]:
            one_band = True
            intersection_points_URB[0] = (0, 0)
            intersection_points_URB[-1] = (0, 0)
        print(f"Do we have a single band? {one_band}")

    return (intersection_points_LRB[0][0], intersection_points_LRB[-1][0]), (intersection_points_URB[0][0], intersection_points_URB[-1][0])

#the metric will be given by the product between the gain and the length of the readout band (for both bands)
def find_metric_np(frequency_array: np.ndarray, 
                   smoothed_profile: np.ndarray, 
                   pump_frequency: float,
                   step: float=16e6, 
                   n_step: int=10, 
                   cut_off: float=11.5e9):
    min_LRB, min_URB = find_gain_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
    LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)

    print(f"Gain in the Lower Band: {min_LRB} dB\nGain in the Upper Band: {min_URB} dB")
    print(f"Length of the Lower Readout Band: {LRB_points[-1]-LRB_points[0]} Hz\nLength of the Upper Readout Band: {URB_points[-1]-URB_points[0]} Hz")

    return min_LRB*(LRB_points[-1]-LRB_points[0]), min_URB*(URB_points[-1]-URB_points[0])

#finds tha max difference between the original array and the smoothed one in the regions of interest
def find_R_max_np(frequency_array: np.ndarray, 
                  profile_array: np.ndarray,
                  smoothed_profile: np.ndarray,
                  pump_frequency: float, 
                  step: float=16e6, 
                  n_step: int=10, 
                  cut_off: float=11.5e9):
    min_LRB, min_URB = find_gain_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
    
    #set the values to 0 if gain = 0
    if min_LRB == 0 and min_URB == 0:
        max_diff_LRB = 0
        max_diff_URB = 0
    elif min_LRB == 0:
        max_diff_LRB = 0
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        
        #Upper band slicing
        URB_test = frequency_array[frequency_array >= URB_points[0]]
        URB_list = URB_test[URB_test <= URB_points[-1]]
    
        lim_dx = profile_array[frequency_array >= URB_list[0]]
        lim_freq_dx = frequency_array[frequency_array >= URB_list[0]]
        lim_URB = lim_dx[lim_freq_dx <= URB_list[-1]]

        lim_sx_avg = smoothed_profile[frequency_array >= URB_list[0]]
        lim_URB_avg = lim_sx_avg[lim_freq_dx <= URB_list[-1]]

        #compute the difference and find the frequency
        max_diff_URB = np.max(np.abs(lim_URB-lim_URB_avg))
        max_index_dx = np.where(np.abs(lim_URB-lim_URB_avg) == max_diff_URB)

    elif min_URB == 0:
        max_diff_URB = 0
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        #Lower band slicing
        LRB_test = frequency_array[frequency_array >= LRB_points[0]]
        LRB_list = LRB_test[LRB_test <= LRB_points[-1]]

        lim_sx = profile_array[frequency_array >= LRB_list[0]]
        lim_freq_sx = frequency_array[frequency_array >= LRB_list[0]]
        lim_LRB = lim_sx[lim_freq_sx <= LRB_list[-1]]

        lim_dx_avg = smoothed_profile[frequency_array >= LRB_list[0]]
        lim_LRB_avg = lim_dx_avg[lim_freq_sx <= LRB_list[-1]]

        max_diff_LRB = np.max(np.abs(lim_LRB-lim_LRB_avg))
        max_index_sx = np.where(np.abs(lim_LRB-lim_LRB_avg) == max_diff_LRB)

    else:
        #same as before but for both bands
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        LRB_test = frequency_array[frequency_array >= LRB_points[0]]
        LRB_list = LRB_test[LRB_test <= LRB_points[-1]]
    
        URB_test = frequency_array[frequency_array >= URB_points[0]]
        URB_list = URB_test[URB_test <= URB_points[-1]]

        lim_sx = profile_array[frequency_array >= LRB_list[0]]
        lim_freq_sx = frequency_array[frequency_array >= LRB_list[0]]
        lim_LRB = lim_sx[lim_freq_sx <= LRB_list[-1]]

        lim_dx_avg = smoothed_profile[frequency_array >= LRB_list[0]]
        lim_LRB_avg = lim_dx_avg[lim_freq_sx <= LRB_list[-1]]
    
        lim_dx = profile_array[frequency_array >= URB_list[0]]
        lim_freq_dx = frequency_array[frequency_array >= URB_list[0]]
        lim_URB = lim_dx[lim_freq_dx <= URB_list[-1]]

        lim_sx_avg = smoothed_profile[frequency_array >= URB_list[0]]
        lim_URB_avg = lim_sx_avg[lim_freq_dx <= URB_list[-1]]

        max_diff_LRB = np.max(np.abs(lim_LRB-lim_LRB_avg))
        max_index_sx = np.where(np.abs(lim_LRB-lim_LRB_avg) == max_diff_LRB)

        max_diff_URB = np.max(np.abs(lim_URB-lim_URB_avg))
        max_index_dx = np.where(np.abs(lim_URB-lim_URB_avg) == max_diff_URB)
        
    print(f'Max difference in the Lower Readout Band: {max_diff_LRB} at Frequency = {LRB_list[max_index_sx][0]} Hz' if max_diff_LRB != 0 else f'Max difference in the Lower Readout Band: {max_diff_LRB}')
    print(f'Max difference in the Upper Readout Band: {max_diff_URB} at Frequency = {URB_list[max_index_dx][0]} Hz' if max_diff_URB != 0 else f'Max difference in the Upper Readout Band: {max_diff_URB}')

    return max_diff_LRB, max_diff_URB

#finds the standard deviation for the two bands
#same logic as find_r_max_np but with std instead of difference
def find_R_avg_np(frequency_array: np.ndarray, 
                  profile_array: np.ndarray, 
                  smoothed_profile: np.ndarray, 
                  pump_frequency: float, 
                  step: float=16e6, 
                  n_step: int=10, 
                  cut_off: float=11.5e9):
    min_LRB, min_URB = find_gain_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
    #set values to 0 if gain = 0
    if min_LRB == 0 and min_URB == 0:
        std_LRB = 0
        std_URB = 0
    elif min_LRB == 0:
        std_LRB = 0
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        URB_test = frequency_array[frequency_array >= URB_points[0]]
        URB_list = URB_test[URB_test <= URB_points[-1]]

        lim_dx = profile_array[frequency_array >= URB_list[0]]
        lim_freq_dx = frequency_array[frequency_array >= URB_list[0]]
        lim_URB = lim_dx[lim_freq_dx <= URB_list[-1]]

        lim_sx_avg = smoothed_profile[frequency_array >= URB_list[0]]
        lim_URB_avg = lim_sx_avg[lim_freq_dx <= URB_list[-1]]

        #find std
        std_URB = np.std((lim_URB-lim_URB_avg))        

    elif min_URB == 0:
        std_URB = 0
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        LRB_test = frequency_array[frequency_array >= LRB_points[0]]
        LRB_list = LRB_test[LRB_test <= LRB_points[-1]]

        lim_sx = profile_array[frequency_array >= LRB_list[0]]
        lim_freq_sx = frequency_array[frequency_array >= LRB_list[0]]
        lim_LRB = lim_sx[lim_freq_sx <= LRB_list[-1]]

        lim_dx_avg = smoothed_profile[frequency_array >= LRB_list[0]]
        lim_LRB_avg = lim_dx_avg[lim_freq_sx <= LRB_list[-1]]

        #find std
        std_LRB = np.std((lim_LRB-lim_LRB_avg))

    else:
        LRB_points, URB_points = find_bandwidth_np(frequency_array, smoothed_profile, pump_frequency, step, n_step, cut_off)
        LRB_test = frequency_array[frequency_array >= LRB_points[0]]
        LRB_list = LRB_test[LRB_test <= LRB_points[-1]]
    
        URB_test = frequency_array[frequency_array >= URB_points[0]]
        URB_list = URB_test[URB_test <= URB_points[-1]]

        lim_sx = profile_array[frequency_array >= LRB_list[0]]
        lim_freq_sx = frequency_array[frequency_array >= LRB_list[0]]
        lim_LRB = lim_sx[lim_freq_sx <= LRB_list[-1]]

        lim_dx_avg = smoothed_profile[frequency_array >= LRB_list[0]]
        lim_LRB_avg = lim_dx_avg[lim_freq_sx <= LRB_list[-1]]
    
        lim_dx = profile_array[frequency_array >= URB_list[0]]
        lim_freq_dx = frequency_array[frequency_array >= URB_list[0]]
        lim_URB = lim_dx[lim_freq_dx <= URB_list[-1]]

        lim_sx_avg = smoothed_profile[frequency_array >= URB_list[0]]
        lim_URB_avg = lim_sx_avg[lim_freq_dx <= URB_list[-1]]

        #find the std in the readout band
        std_LRB = np.std((lim_LRB-lim_LRB_avg))
        std_URB = np.std((lim_URB-lim_URB_avg))

    print(f"std for the Lower Readout Band: {std_LRB}\nstd for the Upper Readout Band: {std_URB}")

    return std_LRB, std_URB


# EXAMPLE FOR ARGO #

S21_argo = xr.open_dataset(r"Data/ARGO SW2311019A - Fine gain tune up.h5", engine='h5netcdf')['S21'].rf.LogMag()
argo_profile = S21_argo.sel({"Pump frequency": 6.33e9})

argo_window, p0, p0_frequency = fft_for_window_np(argo_profile["Frequency"].data, argo_profile.data[0], 6.33e9, perc_off=0.15)

argo_smooth = padded_moving_average_np(argo_profile.data[0], argo_window)

#plot the original profile and the smoothed one
plt.subplots(figsize=[10, 3], dpi=200)

plt.plot(argo_profile[22]["Frequency"].data, argo_profile[0], color="pink", label="Original profile")
plt.plot(argo_profile[22]["Frequency"].data, argo_smooth, color="red", label="Moving average")
plt.grid()
plt.legend()
plt.xlabel("Frequency / Hz")
plt.ylabel("|S21| / dB")
plt.tight_layout()
plt.show()

G_LRB, G_URB = find_gain_np(argo_profile["Frequency"].data, argo_smooth, 6.33e9)

LRB_points, URB_points = find_bandwidth_np(argo_profile["Frequency"].data, argo_smooth, 6.33e9)

metric_LRB, metric_URB = find_metric_np(argo_profile["Frequency"].data, argo_smooth, 6.33e9)

Rmax_L, Rmax_U = find_R_max_np(argo_profile["Frequency"].data, argo_profile.data[0], argo_smooth, 6.33e9)

avg_LRB, avg_URB = find_R_avg_np(argo_profile["Frequency"].data, argo_profile.data[0], argo_smooth, 6.33e9)