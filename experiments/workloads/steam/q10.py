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
    logger = create_logger(log_dir="log/steam", log_name=f"q10_r{ROUND}{ablation_suffix}")
    data = pd.read_csv("testdata/steam_games.csv")
    data["rating"] = nvn.ImageArray(data["rating"])
    df = nvn.DataFrame(data)
    
    logger.info(f"Q10: Compute the average price of games that support both Windows and MacOS and receive a positive review.")
    df.semantic_map(user_instruction="Give the video game a binary review (positive or negative) based on the existing review.", input_column="overall_reviews", output_column="comments")
    df.semantic_filter(user_instruction="The game receives a positive review.", input_column="comments")
    df.semantic_filter(user_instruction="According to the given PEGI rating (in picture), check if the game is only suitable for adults (18 years or older).", input_column="rating")
    df.semantic_filter(user_instruction="The game supports both Windows and MacOS.", input_column="platforms")
    df.semantic_reduce(user_instruction="Compute the average original price.", input_column="original_price")

    config = OptimizeConfig(do_logical_optimization=LO, do_physical_optimization=PO, sample_size=5, improve_margin=0.2, approx_mode=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output, cost, runtime = df.optimize_and_execute(optim_config=config)
    if ROUND == 1:
        output.to_csv(f"workloads/steam_output/q10_out{ablation_suffix}.csv", index=False)
        logger.info(f"saved output to workloads/steam_output/q10_out{ablation_suffix}.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
