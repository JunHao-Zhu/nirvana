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
    api_key="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",
)


if __name__ == "__main__":
    ablation_suffix = ""
    if not LO:
        ablation_suffix += "_wolo"
    if not PO:
        ablation_suffix += "_wopo"
    logger = create_logger(log_dir="log/imdb", log_name=f"q10_r{ROUND}{ablation_suffix}")
    data = pd.read_csv("testdata/imdb_movie_info.csv")
    df = nvn.DataFrame(data)
    
    logger.info(f"Q10: Count the number of crime movies with a rating higher than 8.5 and lower than 9.")
    df.semantic_map(user_instruction="According to the movie plot, extract the genre(s) of each movie.", input_column="Plot", output_column="Genre")
    df.semantic_filter(user_instruction="The rating is higher than 8.5.", input_column="IMDB_rating")
    df.semantic_filter(user_instruction="The rating is lower than 9.", input_column="IMDB_rating")
    df.semantic_filter(user_instruction="The movie belongs to crime movies.", input_column="Genre")
    df.semantic_reduce(user_instruction="Count the number of crime movies.", input_column="Title")

    config = OptimizeConfig(do_logical_optimization=LO, do_physical_optimization=PO, sample_size=5, improve_margin=0.2, approx_mode=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output, cost, runtime = df.optimize_and_execute(optim_config=config)
    if ROUND == 1:
        output.to_csv(f"workloads/imdb_output/q10_out{ablation_suffix}.csv", index=False)
        logger.info(f"saved output to workloads/imdb_output/q10_out{ablation_suffix}.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
