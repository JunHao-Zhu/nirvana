import pandas as pd
import mahjong as mjg


def test_image_data():
    mjg.configure_llm_backbone(
        model_name="Deepseek-chat",
        api_key="<Your-API-Key>",
        base_url="https://api.deepseek.com"
    )

    logo_imgs = mjg.ImageArray([
        "https://spark.apache.org/images/spark-logo.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c6/PyTorch_logo_black.svg/488px-PyTorch_logo_black.svg.png",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Databricks_Logo.png/960px-Databricks_Logo.png"
    ])
    df = pd.DataFrame({
        "names": ["Spark", "Pytorch", "Databricks"],
        "logos": logo_imgs
    })
    
    outputs = mjg.ops.filter(df, "Whether the image is a software logo?", input_schema="logos", strategy="plain_llm")
    return outputs
