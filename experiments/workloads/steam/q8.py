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
    logger = create_logger(log_dir="log/steam", log_name=f"q8_r{ROUND}{ablation_suffix}")
    data = pd.read_csv("testdata/steam_games.csv")
    data["image"] = nvn.ImageArray(data["image"])
    df = nvn.DataFrame(data)
    
    logger.info(f"Q8: Count the number of games that only have one developer with a rating higher than 90.")
    df.semantic_filter(user_instruction="The rating is higher than 90.", input_column="metacriticts")
    df.semantic_filter(user_instruction="Does the video game has only one developer?", input_column="developer")
    df.semantic_reduce(user_instruction="Count the number of games.", input_column="title")

    config = OptimizeConfig(do_logical_optimization=LO, do_physical_optimization=PO, sample_size=5, improve_margin=0.2, approx_mode=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output, cost, runtime = df.optimize_and_execute(optim_config=config)
    if ROUND == 1:
        output.to_csv(f"workloads/steam_output/q8_out{ablation_suffix}.csv", index=False)
        logger.info(f"saved output to workloads/steam_output/q8_out{ablation_suffix}.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
