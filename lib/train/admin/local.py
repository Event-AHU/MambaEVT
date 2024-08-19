class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = 'prj_path'  
        self.tensorboard_dir = self.workspace_dir + '/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks'

        # demo path
        self.eventvot_dir = self.workspace_dir + 'dataset_demo/eventvot'
        self.eventvot_val_dir = self.eventvot_dir

        self.fe108_dir = self.workspace_dir + 'dataset_demo/fe108'
        self.fe108_val_dir = self.fe108_dir

        self.visevent_dir= self.workspace_dir + 'dataset_demo/visevent'
        self.visevent_val_dir=self.visevent_dir