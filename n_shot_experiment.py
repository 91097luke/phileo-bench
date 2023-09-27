import numpy
from datetime import date

import training_script
import experiment_plots

N_SHOTS = [1, 10, 100, 1000, 2500, 5000, 7500, 10000]

parser = training_script.get_args()
args = vars(parser.parse_args())

model_name = args['model_name']
downstream_task = args['downstream_task']
experiment_name = f'trained_models/{date.today().strftime("%d%m%Y")}_{model_name}_{downstream_task}_p_split'

for n_shot in N_SHOTS:
    args['n_shot'] = n_shot
    training_script.main(**args)

experiment_plots.main(folder=experiment_name, plot_title=f'{model_name}_{downstream_task}', y_logscale=True, x_logscale=False)

