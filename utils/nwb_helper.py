from datetime import datetime
from uuid import uuid4
import numpy as np
import scipy.io
import os, glob, json
import pandas as pd
from pynwb import NWBFile
from pynwb.file import Subject
import logging
import pytz
import h5py

def read_names(filename):
    assignment  = filename.split('.')[0].split('-')[1]
    number      = filename.split('.')[0].split('-')[2]
    return np.asarray([assignment, number])


def create_nwb(config, path):

    desired_timezone = pytz.timezone('US/Eastern')

    ################ CREATE NWB FILE WITH METADATA ################################
    ###############################################################################
    nwbfile = NWBFile(
        session_description     = config['session_info']['session_description'],
        identifier              = config['metadata']['identifier'],
        session_start_time      = desired_timezone.localize(config['metadata']['session_start_time']),
        file_create_date        = desired_timezone.localize(config['metadata']['file_create_date']),
        experimenter            = config['general']['lab_info']['experimenter'],
        experiment_description  = config['general']['experiment_info']['experiment_description'],
        session_id              = config['session_info']['session_id'],
        lab                     = config['general']['lab_info']['lab'],                     
        institution             = config['general']['lab_info']['institution'],                                    
        keywords                = config['general']['experiment_info']['keywords'],
        surgery                 = config['general']['experiment_info']['surgery']
    )

    ################ CREATE SUBJECT ################################################
    ################################################################################
    nwbfile.subject = Subject(
        subject_id  = config['subject']['subject_id'],
        date_of_birth= config['subject']['date_of_birth'],
        species     = config['subject']['species'],
        sex         = config['subject']['sex'],
        description = config['subject']['description'],
    )

    ################ CREATE HARDWARE LINKS #########################################
    ################################################################################
    nwbfile.create_device(
        name        = config['hardware']['system_name'], 
        description = config['hardware']['system_description'], 
        manufacturer= config['hardware']['system_manuf']
    )

    nwbfile.create_device(
        name        = config['hardware']['adapter_manuf'], 
        description = config['hardware']['adapter_description'], 
        manufacturer= config['hardware']['adapter_manuf']
    )

    nwbfile.create_device(
        name        = config['hardware']['monitor_name'], 
        description = config['hardware']['monitor_description'], 
        manufacturer= config['hardware']['monitor_manuf']
    )

    nwbfile.create_device(
        name        = config['hardware']['photodiode_name'], 
        description = config['hardware']['photodiode_description'], 
        manufacturer= config['hardware']['photodiode_manuf']
    )
    
    nwbfile.create_device(
        name        = 'Software Used', 
        description = str(['Mworks Client: '+config['software']['mwclient_version'],\
                        'Mworks Server: '+config['software']['mwserver_version'],\
                        'OS: '+config['software']['OS'],\
                        'Intan :'+config['software']['intan_version']])
    )

    ################ CREATE ELECTRODE LINKS ########################################
    ################################################################################
    electrodes = nwbfile.create_device(
        name        = config['hardware']['electrode_name'], 
        description = config['hardware']['electrode_description'], 
        manufacturer= config['hardware']['electrode_manuf']
    )

    all_files = sorted(os.listdir(os.path.join(path, 'SpikeTimes')))
    all_files = [item for item in all_files if not item.startswith('.')]

    
    name_accumulator = []
    for file in all_files:
        name_accumulator.append(read_names(file))
    names = np.vstack(name_accumulator)

    nwbfile.add_electrode_column(name="label", description="label of electrode")
    groups, count_groups = np.unique(names[:,0], return_counts =True)
    ids                  = names[:,1]
    counter              = 0
    # create ElectrodeGroups A, B, C, ..
    for group, count_group in zip(groups, count_groups):
        if len(groups) == 6:
            electrode_description = "Serialnumber: {}. Adapter Version: {}".format(config['array_info']['array_{}'.format(group)]['serialnumber'],\
                            config['array_info']['array_{}'.format(group)]['adapterversion']),
        else: 
            electrode_description = "Serialnumber: {}".format(config['array_info']['array_{}'.format(group)]['serialnumber']),
                
        
        electrode_group = nwbfile.create_electrode_group(
            name        = "group_{}".format(group),
            description = electrode_description[0],
            device      = electrodes,
            location    = 'hemisphere, region, subregion: '+str([config['array_info']['array_{}'.format(group)]['hemisphere'],\
                                config['array_info']['array_{}'.format(group)]['region'],
                                config['array_info']['array_{}'.format(group)]['subregion']]),
            position    = config['array_info']['array_{}'.format(group)]['position']
        )

        # create Electrodes 001, 002, ..., 032 in ElectrodeGroups per channel
        for ichannel in range(count_group):
            nwbfile.add_electrode(
                group       = electrode_group,
                label       = ids[counter],
                location    = 'row, col, elec'+str(json.loads(config['array_info']['intan_electrode_labeling_[row,col,id]'])[counter])
            )
            counter += 1     

    ################ ADD SPIKE TIMES ###############################################
    ################################################################################

    nwbfile.add_unit_column(name="unit", description="millisecond") 

    for filename, i in zip(all_files, range(len(all_files))):
        [assignment, number] = read_names(filename)
        file_path = os.path.join(path, 'SpikeTimes', filename)
        try: data = scipy.io.loadmat(file_path, squeeze_me=True,
                        variable_names='spike_time_ms')['spike_time_ms']
        except:data = scipy.io.loadmat(file_path, squeeze_me=True,
                        variable_names='spikeTime')['spikeTime']
        nwbfile.add_unit(
            spike_times = data, 
            electrodes  = [i],
            electrode_group = nwbfile.electrode_groups[f'group_{assignment}'], 
            unit = 'ms'
        )

    ################ ADD TRIAL TIMES ###############################################
    ################################################################################
    last_spike = data[-1]
    del data
    
    try: 
        [on, off] = config['session_info']['session_description'].split(', ')[3].split(': ')[-1].split("/")
        on_start  = 0
        on_dur    = int(on)
        off_dur   = int(off)

        
        nwbfile.add_trial_column(name="unit", description="millisecond")
        while on_start < last_spike:

            nwbfile.add_trial(
                start_time= float(on_start),
                stop_time = float(on_start+on_dur),
                unit = 'ms')
        
            on_start += on_dur+off_dur
    except: pass 

    ################ ADD PSTH IF AVAIL #############################################
    ################################################################################

    

    if 'psth' in os.listdir(path):
        psthpath = path+'/psth/'+os.listdir(path+'/psth')[0]
        psth = scipy.io.loadmat(psthpath)
        data = psth['psth']
        try: start_time_ms, stop_time_ms, tb_ms = psth['meta'][0][0]
        except: 
            start_time_ms = psth['meta'][0][0][0][0]
            stop_time_ms  = psth['meta'][0][0][1][0]
            tb_ms         = psth['meta'][0][0][2][0]
        meta = [start_time_ms.flatten()[0], stop_time_ms.flatten()[0], tb_ms.flatten()[0]]

        nwbfile.add_scratch(
            data,
            name="psth",
            description="psth [stimuli x reps x timebins x channels]",
            )
        
        nwbfile.add_scratch(
                meta,
                name="psth meta",
                description="start_time_ms, stop_time_ms, tb_ms",
                )

    # if 'h5Files' in os.listdir(path):
    #     if len(os.listdir(os.path.join(path, 'h5Files'))) != 0:

    #         filename = os.listdir(os.path.join(path, 'h5Files'))[0]
    #         file = h5py.File(os.path.join(path, 'h5Files', filename),'r+') 
    #         data = file['psth'][:]
    #         meta = np.array([np.array(file['meta']['start_time_ms']), np.array(file['meta']['stop_time_ms']), np.array(file['meta']['tb_ms'])])
    #         file.close()

    #         nwbfile.add_scratch(
    #             data,
    #             name="psth",
    #             description="psth, uncorrected [stimuli x reps x timebins x channels]",
    #             )
            
    #         nwbfile.add_scratch(
    #                 meta,
    #                 name="psth meta",
    #                 description="start_time_ms, stop_time_ms, tb_ms",
    #                 )

    return nwbfile


def calc_psth(nwbfile, mworks_dir, start_time_ms, stop_time_ms, tb_ms, n_stimuli = None):
    start_time_ms = int(start_time_ms)
    stop_time_ms  = int(stop_time_ms) 
    tb_ms         = int(tb_ms)

    ################ MODIFIED FROM THE SPIKE-TOOLS-CHONG CODE ######################
    ################################################################################

    mwk_data = pd.read_csv(mworks_dir)
    mwk_data = mwk_data[mwk_data.fixation_correct == 1]
    if 'photodiode_on_us' in mwk_data.keys():
        samp_on_ms = np.asarray(mwk_data['photodiode_on_us']) / 1000.
        logging.info('Using photodiode signal for sample on time')
    else:
        samp_on_ms = np.asarray(mwk_data['samp_on_us']) / 1000.
        logging.info('Using MWorks digital signal for sample on time')

    # Load spikeTime file for current channel
    spikeTimes = nwbfile.units[:].spike_times
    # Re-order the psth to image x reps
    max_number_of_reps = max(np.bincount(mwk_data['stimulus_presented']))  # Max reps obtained for any image
    if max_number_of_reps == 0:
        exit()
    mwk_data['stimulus_presented'] = mwk_data['stimulus_presented'].astype(int)  # To avoid indexing errors

    if n_stimuli is None:
            image_numbers = np.unique(mwk_data['stimulus_presented'])  # TODO: if not all images are shown (for eg, exp cut short), you'll have to manually type in total # images
    else:
        image_numbers = np.arange(1,n_stimuli+1) # all of my image starts with #1

    timebase = np.arange(start_time_ms, stop_time_ms, tb_ms)
    PSTH = np.full((len(image_numbers), max_number_of_reps, len(timebase),spikeTimes.shape[0]), np.nan)

    for num in range(spikeTimes.shape[0]):
        spikeTime = np.asanyarray(spikeTimes[num])
        osamp = 10
        psth_bin = np.zeros((len(samp_on_ms), osamp*(stop_time_ms-start_time_ms)))
        psth_matrix = np.full((len(samp_on_ms), len(timebase)), np.nan)

        for i in range(len(samp_on_ms)):

            sidx = np.floor(osamp*(spikeTime[(spikeTime>=(samp_on_ms[i]+start_time_ms))*(spikeTime<(samp_on_ms[i]+stop_time_ms))]-(samp_on_ms[i]+start_time_ms))).astype(int)
            psth_bin[i, sidx] = 1
            psth_matrix[i] = np.sum(np.reshape(psth_bin[i], [len(timebase), osamp*tb_ms]), axis=1)
        
        
        psth = np.full((len(image_numbers), max_number_of_reps, len(timebase)), np.nan)  # Re-ordered PSTH

        for i, image_num in enumerate(image_numbers):
            index_in_table = np.where(mwk_data.stimulus_presented == image_num)[0]
            selected_cells = psth_matrix[index_in_table, :]
            psth[i, :selected_cells.shape[0], :] = selected_cells

        logging.info(psth.shape)
        # Save psth data
        PSTH[:,:,:,num] = psth
        
    meta = {'start_time_ms': start_time_ms, 'stop_time_ms': stop_time_ms, 'tb_ms': tb_ms}
    cmbined_psth = {'psth': PSTH, 'meta': meta}
    return PSTH






def get_psth_from_nwb(nwbfile, path, start_time, stop_time, timebin, n_stimuli=None):

    # Find the MWORKS File
    with open(os.path.join(path, 'NWBInfo.txt')) as f:
        lines = f.readlines()
        line2 = lines[0].split(',')[1]
        ind = line2.split('/').index('intanproc')
        mwk_file = glob.glob(os.path.join('/', *line2.split('/')[0:ind], 'mworksproc', \
                '_'.join(map(str, line2.split('/')[ind+1].split('_')[0:3]))+'*_mwk.csv'))   
        
    assert len(mwk_file) == 1
    mwk_file = mwk_file[0]
    assert os.path.isfile(mwk_file)==True

    ################ MODIFIED FROM THE SPIKE-TOOLS-CHONG CODE ######################
    ################################################################################
    
    mwk_data = pd.read_csv(mwk_file)
    mwk_data = mwk_data[mwk_data.fixation_correct == 1]
    if 'photodiode_on_us' in mwk_data.keys():
        samp_on_ms = np.asarray(mwk_data['photodiode_on_us']) / 1000.
        logging.info('Using photodiode signal for sample on time')
    else:
        samp_on_ms = np.asarray(mwk_data['samp_on_us']) / 1000.
        logging.info('Using MWorks digital signal for sample on time')
    
    # Load spikeTime file for current channel
    spikeTimes = nwbfile.units[:].spike_times
    # Re-order the psth to image x reps
    max_number_of_reps = max(np.bincount(mwk_data['stimulus_presented']))  # Max reps obtained for any image
    if max_number_of_reps == 0:
        exit()
    mwk_data['stimulus_presented'] = mwk_data['stimulus_presented'].astype(int)  # To avoid indexing errors
    
    if n_stimuli is None:
            image_numbers = np.unique(mwk_data['stimulus_presented'])  # TODO: if not all images are shown (for eg, exp cut short), you'll have to manually type in total # images
    else:
        image_numbers = np.arange(1,n_stimuli+1) # all of my image starts with #1

    timebase = np.arange(start_time, stop_time, timebin)
    PSTH = np.full((len(image_numbers), max_number_of_reps, len(timebase),spikeTimes.shape[0]), np.nan)

    for num in range(spikeTimes.shape[0]):
        spikeTime = np.asanyarray(spikeTimes[num])
        osamp = 10
        psth_bin = np.zeros((len(samp_on_ms), osamp*(stop_time-start_time)))
        psth_matrix = np.full((len(samp_on_ms), len(timebase)), np.nan)

        for i in range(len(samp_on_ms)):

            sidx = np.floor(osamp*(spikeTime[(spikeTime>=(samp_on_ms[i]+start_time))*(spikeTime<(samp_on_ms[i]+stop_time))]-(samp_on_ms[i]+start_time))).astype(int)
            psth_bin[i, sidx] = 1
            psth_matrix[i] = np.sum(np.reshape(psth_bin[i], [len(timebase), osamp*timebin]), axis=1)
        
        
        psth = np.full((len(image_numbers), max_number_of_reps, len(timebase)), np.nan)  # Re-ordered PSTH

        for i, image_num in enumerate(image_numbers):
            index_in_table = np.where(mwk_data.stimulus_presented == image_num)[0]
            selected_cells = psth_matrix[index_in_table, :]
            psth[i, :selected_cells.shape[0], :] = selected_cells

        logging.info(psth.shape)
        # Save psth data
        PSTH[:,:,:,num] = psth
        
    meta = {'start_time_ms': start_time, 'stop_time_ms': stop_time, 'tb_ms': timebin}
    cmbined_psth = {'psth': PSTH, 'meta': meta}

    return cmbined_psth
   
