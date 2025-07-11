import os
import io
import time
import logging
import pandas as pd
from PIL import Image
import lotus
from lotus.models import LM
from lotus.types import CascadeArgs

ROUND = 1
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"


def create_logger(log_dir, log_name):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(f"{log_dir}/{log_name}.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    return logger


if __name__ == "__main__":
    logger = create_logger(log_dir="log/estate_lotus", log_name=f"q1_r{ROUND}")
    lm = LM(model="gpt-4.1", max_batch_size=16)
    proxy_lm = LM(model="gpt-4.1-nano", max_batch_size=16)
    lotus.settings.configure(lm=lm, helper_lm=proxy_lm)
    cascade_args = CascadeArgs(recall_target=0.9, precision_target=0.9, sampling_percentage=0.5, failure_probability=0.2)
    df = pd.read_parquet("testdata/multimodal_real_estate.parquet")
    df["image"] = df["image"].apply(lambda x: Image.open(io.BytesIO(x)))
    df["image"] = lotus.ImageArray(df["image"].tolist())
    
    logger.info(f"Q1: Find the house with a yard.")
    start_time = time.time()
    df = df.sem_filter("From the house {image}, whether the house has a yard or not.", cascade_args=cascade_args)
    end_time = time.time()
    runtime = end_time - start_time
    cost = lm.stats.physical_usage.total_cost + proxy_lm.stats.physical_usage.total_cost

    if ROUND == 1:
        df.to_csv("workloads/estate_output/q1_out_lotus.csv", index=False)
        logger.info(f"saved output to workloads/estate_output/q1_out_lotus.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
