import time
import numpy as np
from ml_logger import logger
from tqdm import tqdm

from plan2vec_experiments import RUN
from torch_utils import tslice

logger.configure(log_directory=RUN.server, prefix=RUN.prefix + "/debug-server-upload/test-1")

data = np.ones([14146, 14146], dtype=np.float32)
with logger.SyncContext(max_workers=20):
    for i, chunk in enumerate(tqdm(tslice(data, chunk=1000), desc="uploading data")):
        logger.log_data({i: chunk}, "pairwise_ds.pkl", overwrite=False if i else True)

for chunk in tqdm(logger.iload_pkl("pairwise_ds.pkl")):
    print(chunk.keys())
