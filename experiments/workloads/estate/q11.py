import io
import os
import base64
import logging
import pandas as pd
from PIL import Image
import nirvana as nvn

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


def transform_to_base64(image: bytes):
    image_obj = Image.open(io.BytesIO(image))
    image_obj = image_obj.convert("RGB")
    buffered = io.BytesIO()
    image_obj.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


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
    logger = create_logger(log_dir="log/estate", log_name=f"q11_r{ROUND}{ablation_suffix}")
    data = pd.read_parquet("testdata/multimodal_real_estate.parquet")
    data["image"] = data["image"].apply(lambda x: transform_to_base64(x))
    data["image"] = nvn.ImageArray(data["image"].tolist())
    df = nvn.DataFrame(data)

    logger.info(f"Q11: Compute the lowest price for the estates that has a swimming pool.")
    df.semantic_map(user_instruction="Extract the house price from the detail about the estate.", input_column="Details", output_column="Price")
    df.semantic_map(user_instruction="Extract the amenities from the estate details.", input_column="Details", output_column="Amenities")
    df.semantic_filter(user_instruction="Is there a swimming pool in the estate.", input_column="Amenities")
    df.semantic_reduce(user_instruction="Compute the lowest price for the estates.", input_column="Price")

    config = OptimizeConfig(do_logical_optimization=LO, do_physical_optimization=PO, sample_size=5, improve_margin=0.2, approx_mode=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output, cost, runtime = df.optimize_and_execute(optim_config=config)
    if ROUND == 1:
        output.to_csv(f"workloads/estate_output/q11_out{ablation_suffix}.csv", index=False)
        logger.info(f"saved output to workloads/estate_output/q11_out{ablation_suffix}.csv...")
    logger.info(f"cost: ${cost:.4f}")
    logger.info(f"runtime: {runtime:.4f} sec")
