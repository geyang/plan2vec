from params_proto import Proto, cli_parse
from termcolor import cprint


@cli_parse
class Args:
    data_path = Proto("~/fair/streetlearn/processed-data/manhattan-small",
                      help="path to the processed streetlearn dataset")
    street_view_size = Proto((64, 64), help="image size for the dataset", dtype=tuple)
    street_view_mode = Proto("omni-gray", help="OneOf[`omni-gray`, `ombi-rgb`]")

    lng_lat_correction = Proto(0.75, help="length correction factor for lattitude.")

    latent_dim = Proto(2, help="latent space for the global embedding")

    load_global_metric = "episodeyang/plan2vec/2019/05-23/streetlearn/manhattan-small/gt-neighbor-success/show-goal"


def visualize_global_metric(all_images, lng_lat, global_metric):
    """assume that embedding function is 2D"""
    import torch

    with torch.no_grad():
        xs = torch.tensor(all_images, dtype=torch.float32)
        cs = global_metric(xs)

    print(cs.cpu().numpy())

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(3, 3), dpi=140)
    plt.scatter(*cs.cpu().numpy().T, color="#23aaff", alpha=0.5)
    # plt.xlim(-0.001, 0.001)
    # plt.ylim(-0.001, 0.001)
    plt.show()
    print('done')


def main():
    from termcolor import cprint
    import torch
    import numpy as np
    from ml_logger import logger
    from plan2vec.models.convnets import GlobalMetricConvDeepL2

    Args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if True:  # load local metric
        cprint('loading global metric', "yellow", end="... ")

        global_metric = GlobalMetricConvDeepL2(1, Args.latent_dim).to(Args.device)

        # hard code args for experiment
        Args.model = type(global_metric).__name__
        logger.log_params(Args=vars(Args))

        logger.load_module(global_metric, Args.load_global_metric)
        # global_metric.eval()
        logger.log_text(str(global_metric), filename="models/global_metric.txt")

        cprint('✔done', 'green')

    if True:  # get rope dataset
        cprint('loading environment dataset', "yellow", end="... ")

        # collect sample here
        from streetlearn import StreetLearnDataset
        from os.path import expanduser

        streetlearn = StreetLearnDataset(expanduser(Args.data_path), Args.street_view_size, Args.street_view_mode)
        streetlearn.select_all()
        Args.streetlearn_bbox = streetlearn.bbox

        cprint('✔done', 'green')

    all_images = streetlearn.images[:, None, ...].astype(np.float32) / 255
    all_states = streetlearn.lng_lat

    visualize_global_metric(all_images, all_states, global_metric.encode)


if __name__ == "__main__":
    from plan2vec_experiments import RUN, instr
    from ml_logger import logger

    if not logger.prefix:
        logger.configure(RUN.server, register_experiment=False)

    # exp_prefix = "episodeyang/plan2vec/2019/05-23/streetlearn/manhattan-small/gt-neighbor-success/show-goal"
    # exp_prefix = \
    #     "episodeyang/plan2vec/2019/05-23/streetlearn/manhattan-small/gt-neighbor-success/show-goal/3e-4/02.53/05.578370"
    exp_prefix = "episodeyang/plan2vec/2019/05-23/streetlearn/manhattan-tiny/gt-neighbor-success/2-agents"
    with logger.PrefixContext(exp_prefix):
        weight_paths = sorted(logger.glob("**/global_metric*.pkl"), reverse=True)

    cprint(f"There are {len(weight_paths)} checkpoints.", "green")
    for path in weight_paths:
        Args.load_global_metric = f"/{exp_prefix}/{path}"
        main()
