from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = ''
    # Set your local paths here.

    settings.fe108_path = settings.prj_dir + 'dataset_demo/fe108'
    settings.visevent_path = settings.prj_dir + 'dataset_demo/visevent'
    settings.eventvot_path = settings.prj_dir + 'dataset_demo/eventvot'

    
    settings.result_plot_path = settings.prj_dir + '/output/test/result_plots'
    settings.results_path = settings.prj_dir + '/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = settings.prj_dir + '/output'


    return settings
