import os
import base64
import logging
from typing import Union
from io import BytesIO
from PIL import Image
import pandas as pd
import palimpzest as pz
from palimpzest.core.lib.fields import ImageBase64Field

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


def fetch_image(image: bytes) -> Union[str, None]:
    if pd.isna(image) or image is None:
        return None

    image_obj = Image.open(BytesIO(image))
    image_obj = image_obj.convert("RGB")
    buffered = BytesIO()
    image_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())


if __name__ == "__main__":
    logger = create_logger(log_dir="log/estate_pz", log_name=f"q1_r{ROUND}")
    data = pd.read_parquet("testdata/multimodal_real_estate.parquet")
    data["image"] = data["image"].apply(fetch_image)
    df = pz.Dataset(data)
    df.schema.image = ImageBase64Field(desc=df.schema.image.desc)

    logger.info(f"Q1: Find the house with a yard")
    df = df.sem_filter(_filter="Observed from the house image, whether the house has a yard?", depends_on="image")

    config = pz.QueryProcessorConfig(policy=pz.MinTimeAtFixedQuality(min_quality=0.8), execution_strategy="parallel", max_workers=16, progress=True)
    logger.info(f"Display the optimization config:\n{str(config)}")
    output = df.run(config)
    if ROUND == 1:
        output.to_df().to_csv("workloads/estate_output/q1_out_pz.csv", index=False)
        logger.info(f"saved output to workloads/estate_output/q1_out_pz.csv...")
    logger.info(f"Optimize Cost: ${output.execution_stats.optimization_cost:.4f}")
    logger.info(f"Optimize Time: {output.execution_stats.optimization_time:.4f}s")
    logger.info(f"Execution Cost: ${output.execution_stats.plan_execution_cost:.4f}")
    logger.info(f"Optimize Time: ${output.execution_stats.plan_execution_time:.4f}")
