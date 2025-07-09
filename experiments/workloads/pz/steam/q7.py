import os
import logging
import pandas as pd
import palimpzest as pz
from palimpzest.core.lib.fields import ImageURLField

ROUND = 1
os.environ["OPENAI_API_KEY"] = "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"


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
    logger = create_logger(log_dir="log/steam_pz", log_name=f"q7_r{ROUND}")
    data = pd.read_csv("testdata/steam_games.csv")
    df = pz.Dataset(data)
    df.schema.rating = ImageURLField(df.schema.rating.desc)

    logger.info(f"Q7: Find all shooting games that support Chinese language.")
    new_column = [
        {"name": "genre", "type": str, "desc": "The game's genre."}
    ]
    df = df.sem_add_columns(cols=new_column, depends_on="description", desc="Extract the genre from the brief summary of the game.")
    df = df.sem_filter("The video game is about shooting.", depends_on="genre")
    df = df.sem_filter("Is Chienese one of the supported languages", depends_on="language")

    config = pz.QueryProcessorConfig(policy=pz.MinTime(), execution_strategy="parallel", max_workers=16, progress=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output = df.run(config)
    if ROUND == 1:
        output.to_df().to_csv("workloads/steam_output/q7_out_pz.csv", index=False)
        logger.info(f"saved output to workloads/steam_output/q7_out_pz.csv...")
    logger.info(f"Optimize Cost: ${output.execution_stats.optimization_cost:.4f}")
    logger.info(f"Optimize Time: {output.execution_stats.optimization_time:.4f}s")
    logger.info(f"Execution Cost: ${output.execution_stats.plan_execution_cost:.4f}")
    logger.info(f"Execution Time: {output.execution_stats.plan_execution_time:.4f}s")
