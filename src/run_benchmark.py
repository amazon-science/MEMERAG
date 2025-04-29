# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import pandas as pd
import argparse
import os

from utils import (build_message, load_jinja_template, call_llm_api,
                   parallelize_on_rows, extract_answer_factuality,
                   impute_labels, compute_bacc, concat_df)
from pathlib import Path
from functools import partial

def split_dataset(df):
    """Split the dataset to create a single row per sentence.

    This function explodes the 'answer' column of the input DataFrame,
    creating separate columns for each key in the answer dictionaries.
    This transformation allows for parallel processing of individual sentences
    in the answer.

    Args:
        df (pandas.DataFrame): Input DataFrame containing 'query', 'context',
                               and 'answer' columns. The 'answer' column should
                               contain lists of dictionaries.

    Returns:
        pandas.DataFrame: A new DataFrame with one row per answer, where each
                          key from the original answer dictionaries becomes a
                          separate column.

    Note:
        The original 'answer' column is dropped after the transformation.
    """
    df = df.explode("answer")
    all_keys = set().union(*(d.keys() for d in df['answer']))

    # Create a new column for each key
    for key in all_keys:
        df[key] = df['answer'].apply(lambda x: x.get(key, None))

    # Drop the original column
    df = df.drop('answer', axis=1)

    return df


def load_prompts(sys_prompt_path, task_prompt_path):
    """Load system and task prompts from specified file paths.

    This function reads and loads Jinja templates for system and task prompts
    from the given file paths.

    Args:
        sys_prompt_path (str): File path to the system prompt template.
        task_prompt_path (str): File path to the task prompt template.

    Returns:
        tuple: A tuple containing two items:
            - system_prompt (jinja2.Template): Loaded system prompt template.
            - task_prompt (jinja2.Template): Loaded task prompt template.
    """
    system_prompt = load_jinja_template(sys_prompt_path)
    task_prompt = load_jinja_template(task_prompt_path)
    return system_prompt, task_prompt

def get_model_provider(model_id):
    """Determine the provider/platform for a given model_id.

    This function maps specific model IDs to their corresponding provider platforms
    such as OpenAI, Amazon Bedrock, or VLLM.

    Args:
        model_id (str): The identifier for the model. Supported values are:
                       'gpt4o_mini', 'llama3_2-90b', 'llama3_2-11b', 'qwen_2_5-32B'

    Returns:
        str: The provider platform name. Possible values are:
             'open_ai', 'bedrock', or 'vllm'

    Raises:
        ValueError: If the provided model_id is not recognized or supported.
    """

    if model_id == "gpt4o_mini":
        return "open_ai"
    elif model_id in ("llama3_2-90b", "llama3_2-11b"):
        return "bedrock"
    elif model_id == "qwen_2_5-32B":
        return "vllm"
    raise ValueError("Invalid argument `model_id`")


def get_llm_prediction_with_retries(row, llm_params, model_provider, args, num_retries=6):
    """Attempt to get a valid prediction from an LLM with multiple retries.

    This function calls the LLM API repeatedly until a valid label is extracted
    or the maximum number of retries is reached.

    Args:
        row (dict): A dictionary containing the 'messages' key for LLM input.
        llm_params (dict): Parameters for the LLM API call.
        args (argparse.Namespace): Arguments containing AWS and model configurations.
        num_retries (int, optional): Maximum number of retry attempts. Defaults to 6.

    Returns:
        dict: A dictionary with the following keys:
            - 'pred_label': The extracted label if valid, or an empty string if all retries fail.
            - 'raw_llm_output': The raw output from the LLM API — useful for debugging.

    """

    while num_retries > 0:
        num_retries -= 1
        llm_output = call_llm_api(messages=row["messages"],
                                  param_dict=llm_params,
                                  region_name=args.bedrock_region,
                                  profile=args.aws_profile,
                                  model_id=args.model_id,
                                  model_provider=model_provider)
        extracted_label = extract_answer_factuality(llm_output)

        if extracted_label is not None:
            return {"pred_label": extracted_label, "raw_llm_output": llm_output}

    return {"pred_label": "", "raw_llm_output": llm_output}


def evaluate(df, gnd_col_name, pred_col_name):
    """Evaluate the performance of predictions against ground truth labels.

    This function filters the dataset, imputes missing or invalid predictions,
    and computes the balanced accuracy (BACC) score.

    Args:
        df (pandas.DataFrame): The dataset containing ground truth and prediction columns.
        gnd_col_name (str): Name of the column containing ground truth labels.
        pred_col_name (str): Name of the column containing predicted labels.

    Returns:
        float: The balanced accuracy (BACC) score.

    Notes:
        - Only 'Supported' and 'Not Supported' labels are considered valid.
        - Predictions not in ['Supported', 'Not Supported'] are imputed with
          the opposite of the ground truth label.
    """

    # remove "challenging to determine" labels
    df = df[df[gnd_col_name].isin(["Supported", "Not Supported"])].copy()

    # impute values
    # If the LLMs does not output [`Supported`, `Not Supported`] labels, then
    # we treat the data point as an error and impute the label opposite
    # to that of ground truth.

    df[pred_col_name] = df.apply(lambda row: impute_labels(row, gnd_col_name,
                                                           pred_col_name,
                                                           ["Supported", "Not Supported"]),
                                 axis=1)

    bacc = compute_bacc(df[gnd_col_name].values.tolist(),
                        df[pred_col_name].values.tolist())
    return bacc


def main(args):
    print(f"\n\ninput arguments: {args}\n\n")

    dataset_path = os.path.join("..", "data", args.dataset_name, f"{args.lang}.jsonl")

    print(f"Loading dataset from `{dataset_path}`\n\n")

    dataset = pd.read_json(dataset_path,
        orient='records', encoding="utf-8", lines=True)


    dataset = split_dataset(dataset)

    sys_prompt_template, task_prompt_template = load_prompts(
        Path(args.sys_prompt_path), Path(args.task_prompt_path))

    model_provider = get_model_provider(args.model_id)

    dataset["messages"] = dataset.apply(lambda row: build_message(row, model_provider, sys_prompt_template,
                                                                  task_prompt_template), axis=1)

    llm_params = {"temperature": args.temperature, "top_p": args.top_p, "max_gen_len": 2000}

    func = partial(get_llm_prediction_with_retries,
                   llm_params=llm_params, args=args, model_provider=model_provider)

    result_df = parallelize_on_rows(dataset, func, num_proc=args.num_proc)

    dataset = concat_df(dataset, result_df)

    bacc = evaluate(dataset, "factuality", "pred_label")

    print(
        f"Bacc = {bacc:.4f} for language {args.lang:<5} using model {args.model_id}")

    # Create output directory if it does not exist
    output_dir = os.path.join("..", "output")
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "llm_judged_dataset.csv")
    dataset.to_csv(output_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True, help="Meta evaluation language. "
                        "Valid values are `en`,`es`,`de`,`fr`,`hi`",
                        choices=['en', 'es', 'de', 'fr', 'hi'])
    parser.add_argument("--dataset_name", required=True, help="The dataset you like to use. "
                        "Valid values are `memerag` and `memerag_ext_w_majority_vote`",
                        choices=["memerag", "memerag_ext_w_majority_vote"])

    parser.add_argument("--model_id", required=True, help="Model name",
                        choices=["gpt4o_mini", "llama3_2-90b", "llama3_2-11b", "qwen_2_5-32B"])
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--top_p", default=0.1, type=float)

    parser.add_argument(
        "--bedrock_region", help="[Optional] Specify the region if you are using bedrock model for inference",
        default=None)
    parser.add_argument(
        "--aws_profile", help="[Optional] Specify AWS profile if required",
        default=None)
    parser.add_argument(
        "--num_proc", help="Number of parallel LLM API calls", default=4, type=int)

    parser.add_argument("--num_retries", help="Number of retries", default=6)

    parser.add_argument("--sys_prompt_path", required=True,
                        help="Path to a system prompt file")

    parser.add_argument("--task_prompt_path", required=True,
                        help="path to task prompt file")

    args = parser.parse_args()
    main(args)