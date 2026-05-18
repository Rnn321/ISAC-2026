import argparse
import json
import concurrent.futures
import re
from typing import List, Dict, Any
from openai import OpenAI
import logging
import os

# Configure logging.
level = logging.INFO
os.makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=level,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/commit_gen.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def configure_logging(log_level_name: str):
    global level
    level = getattr(logging, log_level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)

def try_parse_json_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value

def format_prompt_payload_for_logging(provider: str, model: str, system_prompt: str, user_prompt: str) -> str:
    payload = {
        "provider": provider,
        "model": model,
        # Keep system_prompt as the raw string so logging does not change its shape.
        "system_prompt": system_prompt,
        # user_prompt is usually a JSON string; parse it before pretty-printing for debugging.
        "user_prompt": try_parse_json_string(user_prompt),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

OPENAI_COMMIT_MESSAGE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "commit_message_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "commit_message": {
                    "type": "string"
                }
            },
            "required": ["commit_message"],
            "additionalProperties": False
        }
    }
}

# DeepSeek keeps using the original default key fallback.
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
# TODO: fill in your own key.
LEGACY_DEEPSEEK_API_KEY = ""
# The OpenAI branch currently uses OpenRouter's OpenAI-compatible API.
# In other words:
# 1. The SDK is still openai.OpenAI.
# 2. provider is set to openai.
# 3. base_url defaults to OpenRouter.
# 4. The default key is the OpenRouter key configured here.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
# TODO: fill in your own key.
LEGACY_OPENROUTER_API_KEY = ""

class DeepSeekAPIClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = "https://api.deepseek.com/v1",
        system_prompt: str = "",
        provider: str = "deepseek",
        reasoning_effort: str | None = None,
        verbosity: str | None = None,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.model = model
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity
        client_kwargs = {
            "api_key": self.api_key,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self.client = OpenAI(**client_kwargs)

    def call_api(self, user_prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        if system_prompt is None:
            system_prompt = self.system_prompt
        logger.debug(
            "API request payload\n%s",
            format_prompt_payload_for_logging(
                provider=self.provider,
                model=self.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        request_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
        }
        if self.provider == "openai":
            request_kwargs["response_format"] = OPENAI_COMMIT_MESSAGE_RESPONSE_FORMAT
            if self.reasoning_effort:
                request_kwargs["reasoning_effort"] = self.reasoning_effort
            if self.verbosity:
                request_kwargs["verbosity"] = self.verbosity
        else:
            request_kwargs["response_format"] = {
                "type": "json_object"
            }

        response = self.client.chat.completions.create(**request_kwargs)

        return json.loads(response.choices[0].message.content)

def resolve_default_base_url(provider: str) -> str | None:
    if provider == "deepseek":
        return DEEPSEEK_BASE_URL
    if provider == "openai":
        return OPENROUTER_BASE_URL
    return None

def resolve_api_key(args: argparse.Namespace) -> str:
    if args.api_key:
        return args.api_key

    default_env_name = "OPENROUTER_API_KEY" if args.provider == "openai" else "DEEPSEEK_API_KEY"
    env_name = args.api_key_env or default_env_name
    env_value = os.getenv(env_name)
    if env_value:
        return env_value

    if args.provider == "deepseek":
        return LEGACY_DEEPSEEK_API_KEY
    if args.provider == "openai":
        return LEGACY_OPENROUTER_API_KEY

    raise ValueError(
        f"API key not found. Please set {env_name} or pass --api-key."
    )

def read_commits_from_json(file_path: str) -> List[Dict[str, Any]]:
    """Read commit data from a JSON file, including diff and issue fields."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            logger.info(f"Successfully read commit file with {len(data)} records")

            # Return the original data without extra processing.
            return data
    except Exception as e:
        logger.error(f"Failed to read commit file: {e}")
        return []

def get_first_issue_fields(issue_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract common fields from the first issue for experiment reuse."""
    for _, issue_data in issue_summary.items():
        return {
            "issue_title": issue_data.get("issue_title", ""),
            "issue_body": issue_data.get("issue_body", ""),
        }
    return {
        "issue_title": "",
        "issue_body": "",
    }

def process_control_group(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """Process the control group using only the code diff."""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} has an empty diff")
            return None

        # Build input.
        input_data = {
            "code_diff": code_diff
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # Call API.
        logger.info(f"Processing control commit: {commit_sha}")
        api_response = api_client.call_api(user_prompt)

        # Build result.
        task_id = f"Apache_{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": api_client.model,
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"Error while processing control commit {commit_data.get('commit_sha', 'unknown')}: {e}")
        return None

def process_experimental_group(commit_data: Dict[str, Any], api_client: DeepSeekAPIClient) -> Dict[str, Any] | any:
    """Process the experimental group using code diff plus issue title and body."""
    try:
        commit_sha = commit_data['commit_sha']
        repo = commit_data['repo']

        code_diff = commit_data.get('diff')
        if not code_diff:
            logger.error(f"commit {commit_sha} has an empty diff")
            return None

        # Get issue information.
        issue_summary = commit_data.get('issue_summary', {})
        issue_fields = get_first_issue_fields(issue_summary)
        issue_title = issue_fields.get("issue_title", "")
        issue_body = issue_fields.get("issue_body", "")
        if not issue_body or not issue_body.strip():
            logger.error(f"commit {commit_sha} has empty issue information")
            return None

        # Build input.
        input_data = {
            "code_diff": code_diff,
            "issue_title": issue_title,
            "issue_body": issue_body
        }

        user_prompt = json.dumps(input_data, ensure_ascii=False)

        # Call API.
        logger.info(f"Processing experimental commit: {commit_sha}")
        api_response = api_client.call_api(user_prompt)

        # Build result.
        task_id = f"{repo}_{commit_sha}"
        result = {
            "task_id": task_id,
            "model": api_client.model,
            "label": commit_data['message'],
            "pred": api_response.get("commit_message", "")
        }

        return result

    except Exception as e:
        logger.error(f"Error while processing experimental commit {commit_data.get('commit_sha', 'unknown')}: {e}")
        return None

def process_group_with_threadpool(commits: List[Dict[str, Any]], process_func, api_client: DeepSeekAPIClient, max_workers: int = 5) -> List[Dict[str, Any]]:
    """Process a selected group with a thread pool."""
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks.
        future_to_commit = {
            executor.submit(process_func, commit, api_client): commit
            for commit in commits
        }

        # Collect results.
        for future in concurrent.futures.as_completed(future_to_commit):
            commit = future_to_commit[future]
            try:
                result = future.result()
                if result:  # Only add valid results.
                    results.append(result)
                    logger.info(f"Finished processing: {commit['commit_sha']}")
            except Exception as e:
                logger.error(f"Exception while processing commit {commit['commit_sha']}: {e}")

    return results

def save_results_to_jsonl(results: List[Dict[str, Any]], output_file: str):
    """Save results to a JSON file."""
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(results, file, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {output_file}, total records: {len(results)}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def strip_issue_prefix(message: str) -> str:
    if not message:
        return message
    return re.sub(r'^(\[\w+-\d+\]\s*)+', '', message)

def normalize_commit_messages(commits: List[Dict[str, Any]]):
    for commit in commits:
        commit['message'] = strip_issue_prefix(commit.get('message', ''))
        retrieved = commit.get('retrieved')
        if isinstance(retrieved, dict) and retrieved.get('message'):
            retrieved['message'] = strip_issue_prefix(retrieved['message'])

def get_experiment_settings(model: str) -> Dict[str, Dict[str, Any]]:
    default_input_file = "../issue/data_summary/deepseek-reasoner/spark_github_issue.json"

    system_prompt_control = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format) in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    system_prompt_experimental = """You are a developer, and your task is to write a concise commit message based on the code changes (in .diff format), the related issue title, and the related issue body in a commit.
    Output format: A JSON object with a single key "commit_message" containing the concise commit message.
    Example output: {"commit_message": "[SQL] Add null check in wrapperFor (inside HiveInspectors)."}"""

    return {
        "without_issue": {
            "process_func": process_control_group,
            "system_prompt": system_prompt_control,
            "output_file": f"{model}/spark/without_issue.json",
            "input_file": default_input_file,
        },
        "with_full_issue": {
            "process_func": process_experimental_group,
            "system_prompt": system_prompt_experimental,
            "output_file": f"{model}/spark/with_full_issue.json",
            "input_file": default_input_file,
        },
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single DeepSeek experiment.")
    # The OpenAI / ChatGPT path currently defaults to OpenRouter's OpenAI-compatible API.
    # Defaults:
    # provider=openai
    # base_url=https://openrouter.ai/api/v1
    # api_key=the OpenRouter key in this file, overridable by CLI or environment variable
    #
    # Recommended debug example:
    # python deepseek.py --provider openai --model gpt-5.4 --output-root chatgpt-5.4 --experiment with_full_issue --log-level DEBUG --max-workers 1 --limit 1 --reasoning-effort minimal --verbosity low --output-file chatgpt-5.4/spark/_debug_with_full_issue.json
    #
    # Recommended full run example:
    # python deepseek.py --provider openai --model gpt-5.4 --output-root chatgpt-5.4 --experiment with_full_issue --log-level INFO --max-workers 300 --limit 866 --reasoning-effort minimal --verbosity low
    # Debug example for with_full_issue; run one item first to check logs:
    # python deepseek.py --model deepseek-chat --experiment with_full_issue --log-level DEBUG --max-workers 1 --limit 1 --output-file deepseek-chat/spark/_debug_with_full_issue.json
    #
    # Parameter notes:
    # --model deepseek-chat: call the deepseek-chat model.
    # --experiment with_full_issue: run the with_full_issue experiment.
    # --log-level DEBUG: print detailed logs including model / system_prompt / user_prompt.
    # --max-workers 1: recommended for debugging because log order is clearer.
    # --limit 1: run one sample first, then scale to the full run after checking.
    # --output-file ..._debug_with_full_issue.json: store debug output separately.
    # --model: DeepSeek model name, such as deepseek-chat / deepseek-reasoner.
    # --provider:
    # deepseek: keep the original DeepSeek path and defaults.
    # openai: ChatGPT / GPT-5.4 path; still uses the OpenAI SDK, with base_url defaulting to OpenRouter.
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "openai"],
    )
    # --model:
    # Common DeepSeek values: deepseek-chat / deepseek-reasoner.
    # Common OpenAI value: gpt-5.4.
    parser.add_argument("--model", default="deepseek-chat")
    # --output-root:
    # Controls the output root directory.
    # Examples:
    # --model deepseek-chat -> defaults to deepseek-chat/spark/...
    # --model gpt-5.4 -> defaults to gpt-5.4/spark/...
    # To store OpenAI results under chatgpt-5.4/spark/..., pass --output-root chatgpt-5.4.
    parser.add_argument("--output-root", help="Override the default output root directory.")
    # --api-key:
    # Manually specify the key for this run.
    # This has the highest priority and bypasses environment/default keys.
    parser.add_argument("--api-key", help="Override the API key for the current run.")
    # --api-key-env:
    # Choose which environment variable stores the key.
    # Default rules:
    # deepseek -> DEEPSEEK_API_KEY
    # openai  -> OPENROUTER_API_KEY
    # If no environment value is found, fall back to the default key in this file.
    parser.add_argument("--api-key-env", help="Environment variable name that stores the API key.")
    # --base-url:
    # Manually override the API endpoint.
    # Default rules:
    # deepseek -> https://api.deepseek.com/v1
    # openai  -> https://openrouter.ai/api/v1
    # Usually unnecessary unless switching to another OpenAI-compatible gateway.
    parser.add_argument("--base-url", help="Override the API base URL.")
    # --experiment: experiment name to run.
    # Currently only without_issue / with_full_issue are kept.
    parser.add_argument(
        "--experiment",
        default="without_issue",
        choices=sorted(get_experiment_settings("deepseek-chat").keys()),
    )
    # --input-file: manually specify the input JSON path.
    # Usually unnecessary; empty means using the experiment default.
    # To override, pass a path relative to graduate/gen, for example:
    # ../issue/data_summary/deepseek-reasoner/spark_github_issue.json
    parser.add_argument("--input-file", help="Override the default input json file.")
    # --output-file: manually specify the output JSON path.
    # Useful for debugging to avoid overwriting official results.
    parser.add_argument("--output-file", help="Override the default output json file.")
    # --max-workers: number of worker threads.
    # Use 1 for debugging and a larger value such as 300 for full runs.
    parser.add_argument("--max-workers", type=int, default=300)
    # --limit: process only the first N commits.
    # Useful for debugging; for full runs, leave empty or pass the intended full count.
    parser.add_argument("--limit", type=int, help="Only process the first N commits for debugging.")
    # --reasoning-effort:
    # Only applies when provider=openai; controls GPT-5.4 reasoning effort.
    # Choices: none / minimal / low / medium / high.
    # For short commit message generation, start with minimal.
    # If omitted for a gpt-5 model, minimal is filled in automatically.
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high"],
        help="Reasoning effort for OpenAI GPT-5 style models.",
    )
    # --verbosity:
    # Only applies when provider=openai; controls output length.
    # Choices: low / medium / high.
    # Use low for commit messages to avoid overly long outputs.
    # If omitted for a gpt-5 model, low is filled in automatically.
    parser.add_argument(
        "--verbosity",
        choices=["low", "medium", "high"],
        help="Verbosity for OpenAI GPT-5 style models.",
    )
    # --log-level: log level for this run.
    # DEBUG prints model/system_prompt/user_prompt.
    # INFO is cleaner for full runs.
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity for the current run.",
    )
    return parser.parse_args()

def main():
    args = parse_args()
    configure_logging(args.log_level)
    # API key resolution priority:
    # 1. --api-key
    # 2. The environment variable specified by --api-key-env
    # 3. The provider's default environment variable
    # 4. The default key in this file
    api_key = resolve_api_key(args)
    # output_root only affects the output directory, not the actual model name.
    # Example:
    # --model gpt-5.4 --output-root chatgpt-5.4
    # This still calls gpt-5.4, but writes results to chatgpt-5.4/spark/...
    output_root = args.output_root or args.model
    # base_url defaults based on provider:
    # deepseek -> official DeepSeek endpoint
    # openai  -> OpenRouter's OpenAI-compatible endpoint
    base_url = args.base_url or resolve_default_base_url(args.provider)
    reasoning_effort = args.reasoning_effort
    verbosity = args.verbosity
    # For GPT-5 models, use defaults better suited to commit message generation:
    # reasoning_effort=minimal
    # verbosity=low
    if args.provider == "openai" and args.model.startswith("gpt-5"):
        reasoning_effort = reasoning_effort or "minimal"
        verbosity = verbosity or "low"
    # Load experiment defaults first; CLI input/output overrides them.
    settings = get_experiment_settings(output_root)[args.experiment]
    input_file = args.input_file or settings["input_file"]
    output_file = args.output_file or settings["output_file"]

    commits = read_commits_from_json(input_file)
    # If --limit is provided, truncate by limit.
    # Otherwise, DEBUG mode runs only the first two items for easier debugging.
    if args.limit is not None:
        commits = commits[:args.limit]
    elif logger.isEnabledFor(logging.DEBUG):
        commits = commits[0:2]

    if not commits:
        logger.error("Failed to read data")
        return

    normalize_commit_messages(commits)
    logger.info(f"provider: {args.provider}")
    logger.info(f"Running experiment: experiment={args.experiment}, model={args.model}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Successfully loaded {len(commits)} commit records")

    client = DeepSeekAPIClient(
        api_key=api_key,
        system_prompt=settings["system_prompt"],
        model=args.model,
        base_url=base_url,
        provider=args.provider,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    results = process_group_with_threadpool(
        commits=commits,
        process_func=settings["process_func"],
        api_client=client,
        max_workers=args.max_workers
    )

    save_results_to_jsonl(results, output_file)
    logger.info("Experiment completed!")
    logger.info(f"{args.experiment}: {len(results)}/{len(commits)} succeeded")


if __name__ == "__main__":
    main()

