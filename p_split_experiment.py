import numpy
from datetime import date

import training_script
import experiment_plots

parser = training_script.get_args()
args = vars(parser.parse_args())

model_name = args['model_name']
downstream_task = args['downstream_task']
experiment_name = f'{date.today().strftime("%d%m%Y")}_{model_name}_{downstream_task}_p_split'

for p_split in numpy.arange(start=0.1, stop=1.1, step=0.1):
    args['split_ratio'] = p_split
    training_script.main(**args)

experiment_plots.main(folder=experiment_name, plot_title=f'{model_name}_{downstream_task}', y_logscale=True, x_logscale=False)



