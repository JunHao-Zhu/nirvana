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
    logger = create_logger(log_dir="log/imdb_pz", log_name=f"q9_r{ROUND}")
    data = pd.read_csv("testdata/imdb_movie_info.csv")
    df = pz.Dataset(data)
    df.schema.Poster = ImageURLField(df.schema.Poster.desc)

    logger.info(f"Q9: Find the maximum rating of crime movie with a rating higher than 8.5 and lower than 9.")
    new_column = [
        {"name": "Genre", "type": str, "desc": "The genre(s) of the movie."}
    ]
    df = df.sem_add_columns(cols=new_column, depends_on="Plot", desc="According to the movie plot, extract the genre(s) of each movie.")
    df = df.sem_filter(_filter="The rating is higher than 8.5.", depends_on="IMDB_rating")
    df = df.sem_filter(_filter="The rating is lower than 9.", depends_on="IMDB_rating")
    df = df.sem_filter(_filter="The movie belongs to crime movies.", depends_on="Genre")
    df = df.count()

    config = pz.QueryProcessorConfig(policy=pz.MinTimeAtFixedQuality(min_quality=0.8), execution_strategy="parallel", max_workers=16, progress=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output = df.run(config)
    if ROUND == 1:
        output.to_df().to_csv("workloads/imdb_output/q9_out_pz.csv", index=False)
        logger.info(f"saved output to workloads/imdb_output/q9_out_pz.csv...")
    logger.info(f"Optimize Cost: ${output.execution_stats.optimization_cost:.4f}")
    logger.info(f"Optimize Time: {output.execution_stats.optimization_time:.4f}s")
    logger.info(f"Execution Cost: ${output.execution_stats.plan_execution_cost:.4f}")
    logger.info(f"Optimize Time: ${output.execution_stats.plan_execution_time:.4f}")
