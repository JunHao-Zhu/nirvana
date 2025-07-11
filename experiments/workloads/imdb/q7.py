import os
import logging
import nirvana as nvn
import pandas as pd

from nirvana.optim import OptimizeConfig

ROUND = 1
LO = True
PO = True


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


nvn.configure_llm_backbone(
    model_name="gpt-4o-2024-08-06" if LO else "gpt-4.1-2025-04-14", 
    api_key="<YOUR_API_KEY>",
)


if __name__ == "__main__":
    ablation_suffix = ""
    if not LO:
        ablation_suffix += "_wolo"
    if not PO:
        ablation_suffix += "_wopo"
    logger = create_logger(log_dir="log/imdb", log_name=f"q7_r{ROUND}{ablation_suffix}")
    data = pd.read_csv("testdata/imdb_movie_info.csv")
    df = nvn.DataFrame(data)

    logger.info(f"Q7: Find the genre of the highest-rating movie directed by Steven Spielberg.")
    df.semantic_map(user_instruction="According to the movie plot, extract the genre(s) of each movie.", input_column="Plot", output_column="Genre")
    df.semantic_filter(user_instruction="The movie is directed by Steven Spielberg.", input_column="Director")
    df.semantic_reduce(user_instruction="Find the highest rate in the rest movie.", input_column="IMDB_rating")

    config = OptimizeConfig(do_logical_optimization=LO, do_physical_optimization=PO, sample_size=5, improve_margin=0.2, approx_mode=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output, cost, runtime = df.optimize_and_execute(optim_config=config)
    if ROUND == 1:
        output.to_csv(f"workloads/imdb_output/q7_out{ablation_suffix}.csv", index=False)
        logger.info(f"saved output to workloads/imdb_output/q7_out{ablation_suffix}.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
