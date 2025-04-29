# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

# SPDX-License-Identifier: CC-BY-NC-4.0

import time
import re
import botocore
from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAI, RateLimitError, APITimeoutError
from multiprocessing import Pool
from functools import partial
from tqdm.auto import tqdm
from botocore.config import Config
from langchain_aws import ChatBedrock
from pathlib import Path
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import numpy as np


tqdm.pandas()

config = Config(
    read_timeout=900,
    connect_timeout=900,
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

model_id_to_bedrock_id = {"llama3_2-90b": "us.meta.llama3-2-90b-instruct-v1:0",
                          "llama3_2-11b": "us.meta.llama3-2-11b-instruct-v1:0"}

VLLM_SERVER_URL = "http://localhost:8000/v1"


def call_openai_compatible_model(model_id, messages, param_dict, base_url=None):
    """
    Call an OpenAI-compatible model (GPT or VLLM-served models like Qwen or Nemo).

    Args:
        model_id (str): The ID of the model to use.
        messages (list): The messages to send to the model.
        param_dict (dict): Additional parameters for the API call.
        base_url (str, optional): The base URL for the API. If None, uses the default OpenAI URL.

    Returns:
        str: The content of the model's response.
    """
    client_kwargs = {}
    if base_url:
        client_kwargs["base_url"] = base_url
        client_kwargs["api_key"] = "dummy-token"

    client = OpenAI(**client_kwargs)

    llm_output = client.chat.completions.create(
        model=model_id,
        messages=messages,
        **param_dict
    )

    return llm_output.choices[0].message.content


def call_bedrock_model(model_id, messages, param_dict, region_name, profile):
    """Call AWS Bedrock model.

    Args:
        model_id (str): Model identifier.
        messages (list): List of message dictionaries.
        param_dict (dict): Additional parameters for the API call.
        region_name (str): [optional] AWS region name.
        profile (str): [optional] AWS credentials profile name.

    Returns:
        str: Model's response content.
    """
    llm_client = ChatBedrock(
        model_id=model_id_to_bedrock_id[model_id],
        model_kwargs=param_dict,
        config=config,
        credentials_profile_name=profile,
        region_name=region_name
    )
    llm_output = llm_client.invoke(messages)
    return llm_output.content


def call_llm_api(model_provider, model_id, messages, param_dict, region_name="us-west-2", profile=None):
    """Call LLM API with retry logic.

    Attempts to call the appropriate API (OpenAI, local, or Bedrock) based on model_id.
    Implements retry logic for various exceptions.

    Args:
        model_provider (str): String indicating the model provider. It can be one of `open_ai`, `bedrock` and `vllm`.
        messages (list): List of message dictionaries, generated using `build_message` function.
        param_dict (dict): Additional parameters for the API call.
        region_name (str, optional): AWS region. Defaults to "us-west-2".
        profile (str, optional): AWS profile name.

    Returns:
        str: API response or empty string if all retries fail.

    Raises:
        Exception: For unhandled errors.
    """
    sleep_duration = 60  # 1 min
    max_time = 60 * 10  # 10 min
    counter = 0
    init_time = time.perf_counter()

    while counter < 10 and (time.perf_counter() - init_time) < max_time:
        try:
            if model_provider == "open_ai":
                return call_openai_compatible_model(model_id, messages, param_dict)
            elif model_provider == "vllm":
                return call_openai_compatible_model(model_id, messages, param_dict, base_url=VLLM_SERVER_URL)
            elif model_provider == "bedrock":
                return call_bedrock_model(model_id, messages, param_dict, region_name, profile)
            else:
                raise ValueError(f"Invalid value {model_provider} for argument `model_provider`. Valid values are `open_ai`, `bedrock`, `vllm`.")
        except botocore.exceptions.ClientError as error:
            if error.response['Error']['Code'] == ['ModelTimeoutException']:
                time.sleep(sleep_duration)
            elif error.response['Error']['Code'] in ['ThrottlingException', 'TooManyRequestsException']:
                time.sleep(sleep_duration * 2)
            else:
                print(error)
                raise error
        except APITimeoutError:
            time.sleep(sleep_duration)
        except RateLimitError:
            time.sleep(sleep_duration * 2)
        except Exception as e:
            raise e
        counter += 1
    return ""


def build_message(row, model_provider, sys_prompt, task_prompt):
    """Build message for LLM API based on model type.

    Constructs messages using system and task prompts, adapting the format
    based on whether the model is GPT/Qwen or Bedrock.

    Args:
        row (dict): Input data containing 'query', 'context', and 'sentence'.
        model_provider (str): String indicating the model provider. It can be one of `open_ai`, `bedrock` and `vllm`.
        sys_prompt (jinja2.Template): System prompt template.
        task_prompt (jinja2.Template): Task prompt template.

    Returns:
        list: Formatted messages for LLM API call.
    """
    template_dict = {"query": row["query"],
                     "context": row["context"],
                     "answer_segment": row["sentence"]}
    system_prompt_rendered = sys_prompt.render(**template_dict)
    task_prompt_rendered = task_prompt.render(**template_dict)

    if model_provider in ("open_ai", "vllm"):
        messages = [{'role': 'system', 'content': system_prompt_rendered},
                    {'role': 'user', 'content': task_prompt_rendered}]
    elif model_provider == "bedrock":
        messages = [
            SystemMessage(content=system_prompt_rendered),
            HumanMessage(content=task_prompt_rendered)
        ]
    else:
        raise ValueError(f"Invalid value {model_provider} for argument `model_provider`. Valid values are `open_ai`, `bedrock`, `vllm`.")
    return messages


def run_on_subset(func, data_subset):
    return data_subset.progress_apply(func, axis=1)


def parallelize(data, func, num_proc=8):
    """Apply a function to data in parallel.

    Args:
        data (pd.DataFrame): Input data.
        func (callable): Function to apply.
        num_proc (int): Number of processes. Defaults to 8.

    Returns:
        pd.DataFrame: Result of parallel processing.
    """
    #TODO Simplify this with pool.map and chunksize argument.

    if num_proc > len(data):
        num_proc = len(data)
    data_split = np.array_split(data, num_proc)
    pool = Pool(num_proc)
    result = pool.map(func, data_split)
    data = pd.concat(result)
    pool.close()
    pool.join()
    return pd.DataFrame(data.tolist())


def parallelize_on_rows(data, func, num_proc=8):
    return parallelize(data, partial(run_on_subset, func), num_proc)


def load_jinja_template(path: Path):
    return Environment(
        loader=FileSystemLoader(path.parent),
        autoescape=True
    ).get_template(path.name)


def render_template(template, **kwargs):
    return template.render(**kwargs)


def extract_answer_factuality(s):
    """Extract factuality label from a string.

    Searches for 'Supported' or 'Not Supported' within XML tags
    or using fuzzy matching.

    Args:
        s (str): Input string to search.

    Returns:
        str or None: Extracted label or None if not found.
    """
    patterns = [r'<answer>([\s\S]*?)</answer>', r'<rationale>([\s\S]*?)</rationale>']
    for pattern in patterns:
        matches = re.findall(pattern, s)
        if matches:
            for match in matches:
                normalised_match = match.lower().strip()
                if normalised_match == "supported" or normalised_match == "not supported":
                    return "Not Supported" if normalised_match.startswith("not") else "Supported"
        else:
            # try fuzzy extraction
            s = s.lower()
            if "not supported" in s:
                return "Not Supported"
            elif "supported" in s:
                return "Supported"
            return None


def compute_bacc(y_true, y_pred):
    """Compute balanced accuracy score for binary labels.

    Args:
        y_true: List of true labels.
        y_pred: List of predicted labels.

    Returns:
        float: Balanced accuracy score.
    """
    label_mapping = {"Supported": 1, "Not Supported": 0}
    y_true = [label_mapping[x] for x in y_true]
    y_pred = [label_mapping[x] for x in y_pred]
    return balanced_accuracy_score(y_true, y_pred)


def impute_labels(row, gnd_col_name, impute_col_name, valid_values):
    """Impute invalid labels with the opposite of ground truth.

    Args:
        row: DataFrame row.
        gnd_col_name: Ground truth column name.
        impute_col_name: Column to impute.
        valid_values: List of valid label values.

    Returns:
        str: Original or imputed label.
    """
    if row[impute_col_name] not in valid_values:
        # Since meta evaluator is not producing any labels, we want the baseline
        # to be adversarial
        if row[gnd_col_name] == "Supported":
            return "Not Supported"
        else:
            return "Supported"
    return row[impute_col_name]


def concat_df(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Concatenate two DataFrames horizontally.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.

    Returns:
        pd.DataFrame: Concatenated DataFrame.

    Raises:
        AssertionError: If DataFrames have different number of rows.
    """
    assert df1.shape[0] == df2.shape[0]
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    final_df = pd.concat([df1, df2], axis=1)
    return final_df