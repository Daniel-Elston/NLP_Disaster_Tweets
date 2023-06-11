import sys
import os
import yaml


class ConfigLoader:
    def __init__(self, config_path):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        self.actions = {
            'project': self.setup_project,
            'sys': self.setup_sys,
            'data': self.setup_data,
            'subdirs': self.setup_subdirs,
            
            # Pipeline Parameters
            'raw_dtype': self.setup_raw_dtype,
            'true_dtype': self.setup_true_dtype,
            
            'dd_descriptions': self.setup_dd_descriptions,
            
            'pipeline_parameters': self.setup_data_params,
        }
        

    def setup_project(self, config):
        self.project_name = config['name']
        self.project_dir = config['dir']
        sys.path.append(self.project_dir)

    def setup_sys(self, config):
        self.fig_size_m = tuple(config['fig_size_m'])
        self.fig_size_l = tuple(config['fig_size_l'])


    def setup_data(self, config):
        self.raw_data_train = config.get('raw_data_train')
        self.raw_data_test = config.get('raw_data_test')
        
        # self.parquet_train = config.get('c1_train')
        # self.parquet_test = config.get('c1_test')
        
        self.sdo_parq = config.get('sdo_parq')
        self.sdo_pkl = config.get('sdo_pkl')
         
    def setup_subdirs(self, config):
        for subdir in config:
            full_path = os.path.join(self.project_dir, subdir)
            sys.path.append(full_path)
    
    
    # Pipeline Parameters
    def setup_raw_dtype(self, config):
        self.raw_dtype = config

    def setup_true_dtype(self, config):
        self.true_dtype = config
    
    def setup_sdo_pkl(self, config):
        self.sdo_pkl = config

    def setup_dd_descriptions(self, config):
        self.dd_descriptions = config
    
    def setup_data_params(self, config):
        self.drop_cols = config.get('drop_cols')
        self.text_col = config.get('text_col')
        

    def load(self):
        for key, action in self.actions.items():
            if key in self.config:
                action(self.config[key])

