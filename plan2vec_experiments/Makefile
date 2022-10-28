.PHONY:

SHELL=/bin/zsh
prep=source ~/.zprofile && conda activate plan2vec

USERNAME=
HOST=vremote1.vectorinstitute.ai
PASSWORD=
DATA_DIR=/h/bstadie/.plan2vec
LOCAL_DATA_DIR=/Users/ge/

code:
	find . -name "*.py" | grep -v -e archive -e debug | grep local-metric-learning-rate >> Makefile

pull-cached-data:
	sshpass -p ${PASSWORD} rsync -az -e ssh ${USERNAME}@${HOST}:${DATA_DIR} ${LOCAL_DATA_DIR}
	echo "finished downloading data"
	open ~/.plan2vec

# State space 2D Navigation Domains
local-metric:
	${prep} && python ./goal-mass/goal_mass_local_metric.py &
	${prep} && python ./回-maze/回_maze_local_metric.py &
	${prep} && python ./c-maze/c_maze_local_metric.py
local-vis:
	${prep} && python ./goal-mass/local_metric_visualization.py &
	${prep} && python ./c-maze/local_metric_visualization.py &
	${prep} && python ./回-maze/local_metric_visualization.py
state-plan2vec:
	${prep} && python ./goal-mass/goal_mass_plan2vec.py &
	${prep} && python ./c-maze/c_maze_plan2vec.py &
	${prep} && python ./回-maze/回_maze_plan2vec.py


# Image 2D Navigation Domains
local-metric-image:
	${prep} && python ./goal-mass-image/goal_mass_img_local_metric.py &
	${prep} && python ./回-maze-image/回_maze_img_local_metric.py &
	${prep} && python ./c-maze-image/c_maze_img_local_metric.py
local-vis-image:
	${prep} && python ./goal-mass-image/local_metric_img_visualization.py &
	${prep} && python ./回-maze-image/local_metric_img_visualization.py &
	${prep} && python ./c-maze-image/local_metric_img_visualization.py
plan2vec-gt-neighbor-image:
	${prep} && python ./goal-mass-image/plan2vec_tweak/gt_neighbor_3d.py &
	${prep} && python ./goal-mass-image/plan2vec_tweak/gt_neighbor.py &
	${prep} && python ./回-maze-image/plan2vec_tweak/gt_neighbor_3d.py &
	${prep} && python ./回-maze-image/plan2vec_tweak/gt_neighbor.py &
	${prep} && python ./c-maze-image/plan2vec_tweak/gt_neighbor_3d.py &
	${prep} && python ./c-maze-image/plan2vec_tweak/gt_neighbor.py

analysis-local-metric-learning-rate:
	${prep} && python ./analysis/image-maze/local-metric-learning-rate/c_maze_img_local_metric.py
	# NOTE: After the first is finished, run the following.
	# ${prep} && python ./analysis/image-mazelocal-metric-learning-rate/local_metric_img_visualization.py
analysis-local-metric-learning-rate-image:
	${prep} && python ./analysis/local-metric-learning-rate/state-maze/local_metric_visualization.py
	# NOTE: After the first is finished, run the following.
	# ${prep} && python ./analysis/local-metric-learning-rate/state-maze/maze_local_metric.py
