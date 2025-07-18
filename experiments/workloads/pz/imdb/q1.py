import os
import logging
import palimpzest as pz
import pandas as pd

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
    logger = create_logger(log_dir="log/imdb_pz", log_name=f"q1_r{ROUND}")
    data = pd.read_csv("testdata/imdb_movie_info.csv")
    df = pz.Dataset(data)

    logger.info(f"Q1: Extract the genres of all movies.")
    new_column = [
        {"name": "Genre", "type": str, "desc": "The genre(s) of the movie."}
    ]
    df = df.sem_add_columns(cols=new_column, depends_on="Plot", desc="According to the movie plot, extract the genre(s) of each movie.")

    config = pz.QueryProcessorConfig(policy=pz.MinTimeAtFixedQuality(min_quality=0.8), execution_strategy="parallel", max_worker=16, progress=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output = df.run(config)
    if ROUND == 1:
        output.to_df().to_csv("workloads/imdb_output/q1_out_pz.csv", index=False)
        logger.info(f"saved output to workloads/imdb_output/q1_out_pz.csv...")
    logger.info(f"Optimize Cost: ${output.execution_stats.optimization_cost:.4f}")
    logger.info(f"Optimize Time: {output.execution_stats.optimization_time:.4f}s")
    logger.info(f"Execution Cost: ${output.execution_stats.plan_execution_cost:.4f}")
    logger.info(f"Optimize Time: ${output.execution_stats.plan_execution_time:.4f}")
