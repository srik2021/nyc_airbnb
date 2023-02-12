#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, 
exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    logger.info(f"Reading input artifact: {args.input_artifact} from wandb")
    local_path = wandb.use_artifact(args.input_artifact).file()
    df = pd.read_csv(local_path)

    logger.info(f"Successfully read input artifact: {args.input_artifact} from wandb")

    # Drop outliers
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    df.to_csv(args.output_artifact, index=False)

    logger.info(f"Loading cleaned artifact: {args.output_artifact} to wandb")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)
    logger.info(f"Successfully loaded cleaned artifact: {args.output_artifact} to wandb")
    run.finish()

if __name__ == "__main__":
    """
    Reads and parses the input arguments and invokes the go method to perform
    basic cleaning
    """
    parser = argparse.ArgumentParser(description="A very basic cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Name of the input artifact to read from wandb",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the cleaned artifact to publish on wandb",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the cleaned artifact",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum price to use for filtering",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum price to use for filtering",
        required=True
    )


    args = parser.parse_args()

    go(args)
