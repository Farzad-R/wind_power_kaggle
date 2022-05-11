import sys
import os
import pandas as pd
from pyprojroot import here
from pathlib import Path

sys.path.insert(0, os.path.join(here()))
from src.utils.DataUtils import convert_date_to_timestamp
from src.utils import EnvUtils

config = EnvUtils.read_yaml(here("config/transform_clean.yml"))


class Transform:
    def transform_train():
        # read and convert the date column
        df = pd.read_csv(here(config["transform_train"]["input"]))
        df_converted = convert_date_to_timestamp(df)
        # check the output and save the file
        EnvUtils.check_directory(here(config["transform_train"]["output"]))
        output_path = os.path.join(
            here(),
            config["transform_train"]["output"],
            config["transform_train"]["output_filename"],
        )
        print(output_path)
        df_converted.to_parquet(output_path)
        return print("train data is converted and transformed successfuly")

    def transform_test():
        # read and convert the date column
        df = pd.read_csv(here(config["transform_test"]["input"]))
        df_converted = convert_date_to_timestamp(df)
        # check the output and save the file
        EnvUtils.check_directory(here(config["transform_test"]["output"]))
        output = os.path.join(
            here(),
            config["transform_test"]["output"],
            config["transform_test"]["output_filename"],
        )

        df_converted.to_parquet(output)
        return print("test data is converted and transformed successfuly")

    def transform_wind():
        file_list = os.listdir(here(config["transform_wind"]["input"]))
        # check the output and save the file
        EnvUtils.check_directory(here(config["transform_wind"]["output"]))
        print("Transforming wind files ...")
        for file in file_list:
            # read and convert the date column
            df = pd.read_csv(
                os.path.join(here(), config["transform_wind"]["input"], file)
            )
            df_converted = convert_date_to_timestamp(df)
            file_name = Path(file).stem + ".parquet"
            df_converted.to_parquet(
                os.path.join(here(), config["transform_wind"]["output"], file_name)
            )

        return print(f"All files are converted and transformed successfuly")


if __name__ == "__main__":
    Transform.transform_train()
    Transform.transform_test()
    Transform.transform_wind()
