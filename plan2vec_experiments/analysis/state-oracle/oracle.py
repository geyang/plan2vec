from plan2vec.plan2vec.plan2vec_state import DEBUG, Args, main
from plan2vec_experiments import instr, config_charts
import jaynes


def common_config():
    Args.num_epochs = 5000

    Args.lr = 3e-4
    Args.weight_decay = 0

    Args.gamma = 1
    Args.target_update = 0.9
    Args.top_k = None

    Args.n_rollouts = 400

    Args.neighbor_r = 1.
    Args.term_r = 1.
    Args.plan_steps = 1
    Args.H = 50
    Args.r_scale = 0.2

    Args.visualization_interval = 10

    Args.optim_epochs = 32


def set_goal_mass():
    Args.env_id = 'GoalMassDiscreteIdLess-v0'
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/goal-mass/goal_mass_local_metric/20.47/21.907891"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def set_回_maze():
    Args.env_id = '回MazeDiscreteIdLess-v0'
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/回-maze/回_maze_local_metric/20.47/27.023292"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def set_c_maze():
    Args.env_id = "CMazeDiscreteIdLess-v0"
    local_metric_exp_path = "geyang/plan2vec/2019/06-21/c-maze/c_maze_local_metric/20.47/19.877318"
    Args.load_local_metric = f"/{local_metric_exp_path}/models/local_metric.pkl"


def plan2vec_oracle_eps_greedy(prefix):
    from ml_logger import logger

    Args.num_epochs = 100
    Args.binary_reward = None
    DEBUG.supervised_value_fn = False
    DEBUG.oracle_eps_greedy = True
    DEBUG.random_policy = False
    DEBUG.real_r_distance = False
    _ = instr(main, __postfix=f"oracle_planner/{prefix}", **vars(Args), _DEBUG=vars(DEBUG))
    config_charts(path="oracle_planner.charts.yml")
    logger.log_text("""
    # Oracle Planner
    
    The planning baseline with a greedy, oracle planner
    """, filename="README.md")
    jaynes.run(_)


if __name__ == "__main__":

    jaynes.config("vector", runner=dict(n_cpu=20))

    common_config()

    Args.lr = 1e-4
    Args.eps_greedy = 0.05

    for fn in [set_goal_mass, set_回_maze, set_c_maze]:
        fn()
        plan2vec_oracle_eps_greedy(f"{Args.env_id}")

    jaynes.listen()
