from dateutil.tz import tzlocal
from datetime import datetime
from uuid import uuid4
import yaml
import os
import json
import numpy as np
from ruamel.yaml import YAML
from utils.rhd_helper import read_header
import configparser
import pandas as pd


def load_array_metadata(array_path):
    array          = open(array_path)
    data           = json.load(array)
    
    bank_assignment     = np.unique(np.array(list(data['bank'].values())))              # unique for each group
    indices             = []
    for bank in bank_assignment:
        indices.append(np.where(np.array(list(data['bank'].values()))==bank)[0][0])

    num_arrays          = len(bank_assignment)
    subregions          = np.array(list(data['subregion'].values()))[indices]         # unique for each group
    region              = np.array(list(data['region'].values()))[indices]            # same for all
    hemispheres         = np.array(list(data['hemisphere'].values()))[indices]        # same for all
    serialnumbers       = np.array(list(data['arr'].values()))[indices]               # same for all

    positions           = np.zeros((len(list(data['subregion'].values())), 3))          # row, col, number
    positions[:,0]      = np.array(list(data['row'].values()))
    positions[:,1]      = np.array(list(data['col'].values()))
    positions[:,2]      = np.array(list(data['elec'].values()))

    if num_arrays ==6:
        adapter_versions    = np.array(list(data['adapter_version'].values()))[indices]
        return subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays, adapter_versions
    
    return subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays
       
def load_rec_info(path):
    with open(os.path.join(path,"RecInfo.yaml") , "r") as f:
        return yaml.load(f, Loader = yaml.FullLoader)

def load_intan_info(Rec_Info):
    intan_info      = Rec_Info['intanproc'].split('/')[:-1]
    intan_info[-2]  = 'intanraw'
    intan_info_path = os.path.join('/', *intan_info, 'info.rhd')
    fid = open(intan_info_path, 'rb')
    return read_header(fid)

def load_excel_sarah():
    df_sarah = pd.read_excel( '/braintree/home/aliya277/dandi_brainscore/Pipeline Monkey Schedule New.xlsx' ,sheet_name='pico' )
    new_header = df_sarah.iloc[0]  
    df_sarah = df_sarah[1:]       
    df_sarah.columns = new_header  
    df_sarah = df_sarah.fillna('empty') 
    return df_sarah

def create_yaml(storage_dir, imageset, subj, date, time, array_metadata_path, df = None, adapter_info_avail=False):
    
    if imageset == 'normalizers':
        directory = f'norm_FOSS.sub_pico.{date}_{time}.proc'
    elif imageset == 'normalizers-HVM':
        directory = f'norm_HVM.sub_pico.{date}_{time}.proc'
    else: 
        directory = f"exp_{imageset}.sub_{subj}.{date}_{time}.proc"

    imagesetdir = os.path.join(storage_dir, ".".join(directory.split(".")[0:1]))
    subjectdir  = os.path.join(storage_dir, imagesetdir, ".".join(directory.split(".")[0:2]))
    subjectdir_date  = os.path.join(subjectdir, ".".join(directory.split(".")[0:2])+'.'+date)

    config_dict = dict()

    config_dict['subject'] = {
                'subject_id':   subj,
                'date_of_birth': datetime(2014, 6, 22, tzinfo = tzlocal()), 
                'sex':          'M',
                'species':      'Macaca mulatta',
                'description':  'monkey'
                }
    
    config_dict['general'] ={
                'experiment_info': {
                    'experiment_description': f'Task: Rapid serial visual presentation (RSVP).',
                    'keywords': ['Vistual Stimuli', 'Object Recognition', 'Inferior temporal cortex (IT)', 'Ventral visual pathway'],
                    'surgery' : '3x Utah Array Implant + Headpost',
                    },
                'lab_info': {
                    'university':  'Massachusetts Institute of Technology',
                    'institution': 'McGovern Institute for Brain Research',
                    'lab': 'DiCarlo',
                    'experimenter': 'Goulding, Sarah',
                    }
                }

    config_dict['hardware'] = {
                    'electrode_name': 'Electrode',
                    'electrode_description': 'Utah CerePort Array with a single electrode array',
                    'electrode_manuf': '2020 Blackrock Microsystems, LLC',
                    'system_name': 'RecordingSystem',
                    'system_description': 'RHD Recording System',
                    'system_manuf': '2010-2023 Intan Technologies', 
                    'adapter_name': 'Utah Array Pedestal Connector',
                    'adapter_description': 'Connects Utah Pedestal to Intan RHD Recording System',
                    'adapter_manuf': 'Ripple Neuro', 
                    'photodiode_name': 'DET36A2 Biased Si Detector',
                    'photodiode_description': 'Photodiode for detecting image presentation times. Comes with DET2A Power Adapter',
                    'photodiode_manuf': 'Thorlabs Inc.', 
                    'monitor_name': 'LG UltraGear',
                    'monitor_description': 'LG 32GP850-B 32 UltraGear QHD (2560 x 1440) Nano IPS Gaming Monitor w/ 1ms (GtG) Response Time & 165Hz Refresh Rate.\
                        Manually color calibrated and set to 120 HZ refresh rate.',
                    'monitor_manuf': 'LG', 
                    }
    
    config_dict['software'] ={
                    'mwclient_version': 'Version 0.11 (2022.02.15)',
                    'mwserver_version': 'Version 0.11 (2022.02.15)',
                    'OS': 'macOS Monterery on MAC Pro (Late 2013)',
                    'intan_version': 'Version 3.1.0'
                    }
   
                
    config_dict['metadata'] = {
                    'nwb_version' : '2.6.0',
                    'file_create_date': datetime.strptime(date, "%Y%m%d"),
                    'identifier': str(uuid4()), 
                    'session_start_time':datetime.strptime(date+time, "%Y%m%d%H%M%S")
                    }

    if type(df)==pd.core.series.Series:

        config_dict['paths'] = {
                    'SpikeTime': df['Path: SpikeTimes'],
                    'psth': df['Path: psth'],
                    'h5': df['Path: h5'],
                    }

        df_sarah = load_excel_sarah()
        text = 'Recording Information'
        try:
            rec_info = df_sarah.iloc[int(df['(excel) Index'])-3][2:14]
            for ele in (str(rec_info).split('\n')[1:-1]):
                try:  text += f", {ele.split(' ')[0]} : {ele.split(' ')[-1]}"
                except: pass
        except:pass

        config_dict["session_info"] = {
                    'session_id': directory,
                    'session_description': text,
                    }
    else:
        config_dict["session_info"] = {
                'session_id': directory,
                'session_description': 'RSVP task',
                }
        
    config_dict['PSTH info'] = {}

    psth_info = configparser.ConfigParser()
    psth_info.read('/braintree/data2/active/users/sgouldin/spike-tools-chong/spike_tools/spike_config.ini')
    for option in psth_info.options('PSTH'):
        config_dict['PSTH info'][option] = psth_info.get('PSTH', option)

    config_dict['Filtering info'] = {}

    for option in psth_info.options('Filtering'):
        config_dict['Filtering info'][option] = psth_info.get('Filtering', option)
                
    
    if adapter_info_avail == True:
        subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays, adapter_versions = load_array_metadata(array_metadata_path)
    else:
        subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays = load_array_metadata(array_metadata_path)
    
    config_dict['array_info'] = {'intan_electrode_labeling_[row,col,id]':json.dumps(positions.tolist())}

    if num_arrays == 6:
        for arr, i in zip(bank_assignment, range(num_arrays)) :
                config_dict['array_info']['array_{}'.format(arr)] = {
                            'position': [0.0,0.0,0.0],
                            'serialnumber':     str(serialnumbers[i]),
                            'adapterversion':   str(adapter_versions[i]),
                            'hemisphere':       str(hemispheres[i]),
                            'region':           str(region[i]),
                            'subregion':        str(subregions[i]),
                            }  
    else:
         for arr, i in zip(bank_assignment, range(num_arrays)) :
                config_dict['array_info']['array_{}'.format(arr)] = {
                            'position': [0.0,0.0,0.0],
                            'serialnumber':     str(serialnumbers[i]),
                            'hemisphere':       str(hemispheres[i]),
                            'region':           str(region[i]),
                            'subregion':        str(subregions[i]),
                            }  

    yaml = YAML()
    with open(os.path.join(subjectdir_date,directory,f"config_nwb.yaml"), 'w') as yamlfile:
        yaml.dump((config_dict), yamlfile)




def create_yaml_old(dir_path, array_metadata_path, adapter_info_avail=False):
    
    rec_info = load_rec_info(dir_path)

    config_dict = dict()

    config_dict['subject'] = {
                'subject_id':   'pico',
                'date_of_birth': datetime(2014, 6, 22, tzinfo = tzlocal()), 
                'sex':          'M',
                'species':      'Macaca mulatta',
                'description':  'monkey'
                }
    
    text = 'Recording Information'
    for item, value in zip(list(rec_info)[2:8], list(rec_info.values())[2:8]):
        text += f', {item} : {value}'

    config_dict["session_info"] = {
                'session_id': rec_info['SessionDate']+'_'+rec_info['Stimuli'],
                'session_description': text,
                }
    
    config_dict['SpikeTime path'] = {
                    'intanproc': rec_info['intanproc'],
                    }
    
    config_dict['general'] ={
                'experiment_info': {
                    'experiment_description': f'Task: Rapid serial visual presentation (RSVP).',
                    'keywords': ['Vistual Stimuli', 'Object Recognition', 'Inferior temporal cortex (IT)', 'Ventral visual pathway'],
                    'surgery' : '3x Utah Array Implant + Headpost',
                    },
                'lab_info': {
                    'university':  'Massachusetts Institute of Technology',
                    'institution': 'McGovern Institute for Brain Research',
                    'lab': 'DiCarlo',
                    'experimenter': 'Goulding, Sarah',
                    }
                }

    config_dict['hardware'] = {
                    'electrode_name': 'Electrode',
                    'electrode_description': 'Utah CerePort Array with a single electrode array',
                    'electrode_manuf': '2020 Blackrock Microsystems, LLC',
                    'system_name': 'RecordingSystem',
                    'system_description': 'RHD Recording System',
                    'system_manuf': '2010-2023 Intan Technologies', 
                    'adapter_name': 'Utah Array Pedestal Connector',
                    'adapter_description': 'Connects Utah Pedestal to Intan RHD Recording System',
                    'adapter_manuf': 'Ripple Neuro', 
                    'photodiode_name': 'DET36A2 Biased Si Detector',
                    'photodiode_description': 'Photodiode for detecting image presentation times. Comes with DET2A Power Adapter',
                    'photodiode_manuf': 'Thorlabs Inc.', 
                    'monitor_name': 'LG UltraGear',
                    'monitor_description': 'LG 32GP850-B 32 UltraGear QHD (2560 x 1440) Nano IPS Gaming Monitor w/ 1ms (GtG) Response Time & 165Hz Refresh Rate.\
                        Manually color calibrated and set to 120 HZ refresh rate.',
                    'monitor_manuf': 'LG', 
                    }
    
    config_dict['software'] ={
                    'mwclient_version': 'Version 0.11 (2022.02.15)',
                    'mwserver_version': 'Version 0.11 (2022.02.15)',
                    'OS': 'macOS Monterery on MAC Pro (Late 2013)',
                    'intan_version': 'Version 3.1.0'
                    }
   
                
    config_dict['metadata'] = {
                    'nwb_version' : '2.6.0',
                    'file_create_date': datetime.strptime(rec_info['SessionDate'], "%Y%m%d"),
                    'identifier': str(uuid4()), 
                    'session_start_time':datetime.strptime(rec_info['SessionDate']+rec_info['SessionTime'], "%Y%m%d%H%M%S")
                    }

    config_dict['PSTH info'] = {}

    psth_info = configparser.ConfigParser()
    psth_info.read('/braintree/data2/active/users/sgouldin/spike-tools-chong/spike_tools/spike_config.ini')
    for option in psth_info.options('PSTH'):
        config_dict['PSTH info'][option] = psth_info.get('PSTH', option)

    config_dict['Filtering info'] = {}

    for option in psth_info.options('Filtering'):
        config_dict['Filtering info'][option] = psth_info.get('Filtering', option)
                
    
    if adapter_info_avail == True:
        subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays, adapter_versions = load_array_metadata(array_metadata_path)
    else:
        subregions, hemispheres, region, serialnumbers, bank_assignment, positions, num_arrays = load_array_metadata(array_metadata_path)
    
    config_dict['array_info'] = {'intan_electrode_labeling_[row,col,id]':json.dumps(positions.tolist())}

    if num_arrays == 6:
        for arr, i in zip(bank_assignment, range(num_arrays)) :
                config_dict['array_info']['array_{}'.format(arr)] = {
                            'position': [0.0,0.0,0.0],
                            'serialnumber':     str(serialnumbers[i]),
                            'adapterversion':   str(adapter_versions[i]),
                            'hemisphere':       str(hemispheres[i]),
                            'region':           str(region[i]),
                            'subregion':        str(subregions[i]),
                            }  
    else:
         for arr, i in zip(bank_assignment, range(num_arrays)) :
                config_dict['array_info']['array_{}'.format(arr)] = {
                            'position': [0.0,0.0,0.0],
                            'serialnumber':     str(serialnumbers[i]),
                            'hemisphere':       str(hemispheres[i]),
                            'region':           str(region[i]),
                            'subregion':        str(subregions[i]),
                            }  

    yaml = YAML()
    with open(os.path.join(dir_path,f"config_nwb.yaml"), 'w') as yamlfile:
        yaml.dump((config_dict), yamlfile)
