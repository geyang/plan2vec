from tqdm import tqdm

from ml_logger import logger

logger.configure("http://localhost:8090",
                 "geyang/plan2vec/2019/07-17/streetlearn/plan2vec/gt_success_L1-2")
# logger.configure()
charts = logger.glob('**/img_maze_local_metric.charts.yml')
print(*charts, sep="\n")
for path in tqdm(charts):
    text = logger.load_text(path)
    logger.log_text("""
                    keys:
                      - Args.data_path
                      - Args.global_metric
                      - Args.binary_reward
                      - Args.term_r
                      - DEBUG.ground_truth_neighbor_r
                    charts:
                      - type: file
                        glob: "**/*.png"
                      - type: file
                        glob: "debug/score_*.png"
                      - yKey: success_rate
                        xKey: epoch
                    """
                    , path, dedent=True, overwrite=True)

print('done')
