import requests
import pandas as pd
import argparse
import concurrent.futures
import json
import time
import re
import threading
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import os
from dotenv import load_dotenv
import logging

GITHUB = "github"
JIRA = "jira"
SPARKURL = "https://github.com/apache/spark.git"

# Jira project key mapping for full.jsonl cross-repo crawling.
# Do not rely on repo.upper() because many Apache projects historically used different Jira keys,
# e.g., doris -> PALO, predictionio -> PIO, incubator-weex -> WEEX.
REPO_JIRA_KEYS = {
    "spark": ["SPARK"],
    "camel": ["CAMEL"],
    "flink": ["FLINK"],
    "beam": ["BEAM"],
    "airflow": ["AIRFLOW"],
    "doris": ["PALO", "DORIS"],
    "hadoop": ["HADOOP", "HDFS", "YARN", "MAPREDUCE"],
    "kafka": ["KAFKA"],
    "hive": ["HIVE"],
    "mesos": ["MESOS"],
    "pinot": ["PINOT", "THIRDEYE"],
    "hbase": ["HBASE"],
    "iotdb": ["IOTDB"],
    "arrow": ["ARROW", "PARQUET"],
    "incubator-weex": ["WEEX"],
    "datafusion": ["ARROW"],
    "hudi": ["HUDI"],
    "cassandra": ["CASSANDRA"],
    "incubator-kie-drools": ["DROOLS", "JBPM"],
    "mxnet": ["MXNET"],
    "dolphinscheduler": ["DS"],
    "zeppelin": ["ZEPPELIN"],
    "rocketmq": ["ROCKETMQ"],
    "dubbo": ["DUBBO"],
    "couchdb": ["COUCHDB"],
    "storm": ["STORM"],
    "thrift": ["THRIFT"],
    "zookeeper": ["ZOOKEEPER"],
    "flink-cdc": ["FLINK"],
    "predictionio": ["PIO"],
}

# These repos look more like GitHub / Bugzilla / proposal flows in current data; avoid Jira.
NO_JIRA_REPOS = {
    "shardingsphere",
    "pulsar",
    "superset",
    "tvm",
    "tomcat",
    "druid",
    "jmeter",
    "iceberg",
    "echarts",
    "seatunnel",
    "skywalking",
    "apisix",
    "openwhisk",
    "shenyu",
    "answer",
    "incubator-seata",
    "hertzbeat",
    "brpc",
    "shardingsphere-elasticjob",
    "dubbo-spring-boot-project",
}

class ApacheCommitIssueCrawler:
    def __init__(
        self,
        github_token: str = None,
        github_tokens: List[str] = None,
        jira_cookie: str = None,
        operated_count: int = 0,
        preferred_platform: str = JIRA,
        log_level: str = "INFO",
    ):
        load_dotenv()
        env_github_tokens = []
        if os.getenv('GITHUB_TOKENS'):
            env_github_tokens.extend([token.strip() for token in os.getenv('GITHUB_TOKENS').split(',') if token.strip()])
        if os.getenv('GITHUB_TOKEN'):
            env_github_tokens.append(os.getenv('GITHUB_TOKEN').strip())

        explicit_tokens = []
        if github_tokens:
            explicit_tokens.extend([token.strip() for token in github_tokens if token and token.strip()])
        if github_token and github_token.strip():
            explicit_tokens.insert(0, github_token.strip())

        merged_tokens = explicit_tokens + env_github_tokens
        self.github_tokens = list(dict.fromkeys(merged_tokens))
        self.current_github_token_index = 0
        self.github_token = self.github_tokens[0] if self.github_tokens else None
        self.jira_cookie = jira_cookie

        self.github_session = requests.Session()
        github_headers = {
            'Accept': 'application/vnd.github.v3+json',
            'User-Agent': 'Commit-Issue-Crawler/1.0'
        }
        if self.github_token:
            github_headers['Authorization'] = f'token {self.github_token}'
        self.github_session.headers.update(github_headers)

        self.jira_session = requests.Session()
        jira_headers = {
            # Jira auth options (choose one)
            # 1. Basic Auth (username + API Token)
            # 'Authorization': f'Basic {base64.b64encode(f"{username}:{api_token}".encode()).decode()}',
            # or 2. Bearer Token (OAuth)
            # 'Authorization': f'Bearer {jira_token}',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'User-Agent': 'Jira-Crawler/1.0',
        }
        if self.jira_cookie:
            jira_headers['cookie'] = self.jira_cookie
        self.jira_session.headers.update(jira_headers)

        # Default platform priority for repos without a special config.
        # Example: if set to Jira, all repos try Jira first, then fall back to GitHub.
        self.default_preferred_platform = preferred_platform or JIRA
        self.config = {
            SPARKURL: self.default_preferred_platform
        }
        logs_dir = os.path.join(os.path.dirname(__file__), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        resolved_log_level = getattr(logging, (log_level or "INFO").upper(), logging.INFO)
        # Configure logging
        logging.basicConfig(
            level=resolved_log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(logs_dir, 'crawler.log'), encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Rate limit control
        self.request_count = 0
        self.reset_time = time.time()
        self.operated_count = operated_count
        self.github_rate_limit_lock = threading.Lock()
        self.github_token_lock = threading.Lock()
        self.quality_stats_lock = threading.Lock()
        self.stage_audit_lock = threading.Lock()
        self.request_timeout = 30
        self.max_retry_attempts = 5
        self.retry_backoff_cap = 60
        self.quality_stats = self.create_quality_stats()
        self.last_quality_stats_snapshot = self.create_quality_stats()
        self.quality_stats_enabled = False
        self.stage_audit_records = self.create_stage_audit_records()
        self.thread_local = threading.local()

    @staticmethod
    def create_quality_stats() -> Dict[str, int]:
        return {
            'commits_total': 0,
            'source_commit_unavailable': 0,
            'stage1_any_candidate': 0,
            'stage1_no_candidate': 0,
            'stage2_unique_candidate': 0,
            'stage2_multi_candidate_only': 0,
            'stage3_basic_title_filtered': 0,
            'stage3_no_valid_issue': 0,
            'issue_filter_github_pr': 0,
            'issue_filter_body_empty': 0,
            'issue_filter_body_url': 0,
            'stage3_content_kept': 0,
            'final_kept': 0,
        }

    @staticmethod
    def create_stage_audit_records() -> Dict[str, List[Dict]]:
        return {
            'stage1_no_candidate': [],
            'stage2_multi_candidate_only': [],
            'stage3_content_filtered': [],
        }

    def reset_quality_stats(self):
        with self.quality_stats_lock:
            self.quality_stats = self.create_quality_stats()
            self.last_quality_stats_snapshot = self.create_quality_stats()

    def reset_stage_audit_records(self):
        with self.stage_audit_lock:
            self.stage_audit_records = self.create_stage_audit_records()

    def increment_quality_stat(self, key: str, amount: int = 1):
        if not self.quality_stats_enabled:
            return
        with self.quality_stats_lock:
            self.quality_stats[key] = self.quality_stats.get(key, 0) + amount

    def snapshot_quality_stats(self) -> Dict[str, int]:
        with self.quality_stats_lock:
            return dict(self.quality_stats)

    def record_stage_audit(self, stage_key: str, record: Dict):
        if not self.quality_stats_enabled:
            return
        with self.stage_audit_lock:
            self.stage_audit_records.setdefault(stage_key, []).append(record)

    def snapshot_stage_audit_records(self) -> Dict[str, List[Dict]]:
        with self.stage_audit_lock:
            return {key: list(value) for key, value in self.stage_audit_records.items()}

    def set_last_issue_reject_reason(self, reason: Optional[str], details: Optional[Dict] = None):
        self.thread_local.last_issue_reject_reason = reason
        self.thread_local.last_issue_reject_details = details or {}

    def consume_last_issue_reject_reason(self) -> Tuple[Optional[str], Dict]:
        reason = getattr(self.thread_local, 'last_issue_reject_reason', None)
        details = getattr(self.thread_local, 'last_issue_reject_details', {}) or {}
        self.thread_local.last_issue_reject_reason = None
        self.thread_local.last_issue_reject_details = {}
        return reason, details

    @staticmethod
    def merge_quality_stats(base: Dict[str, int], extra: Dict[str, int]) -> Dict[str, int]:
        merged = dict(base)
        for key, value in extra.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    def log_quality_summary(self, summary: Dict[str, int], prefix: str = "Quality summary"):
        total_input = summary.get('commits_total', 0)
        stage1_kept = summary.get('stage1_any_candidate', 0)
        stage1_filtered = max(total_input - stage1_kept, 0)
        stage2_kept = summary.get('stage2_unique_candidate', 0)
        stage2_filtered = max(stage1_kept - stage2_kept, 0)
        stage3_content_kept = summary.get('stage3_content_kept', 0)
        stage3_content_filtered = max(stage2_kept - stage3_content_kept, 0)
        final_kept = summary.get('final_kept', 0)
        stage3_filtered = max(stage2_kept - final_kept, 0)

        self.logger.info(
            f"{prefix}: input_commits={total_input}"
        )
        self.logger.info(
            f"{prefix} - Stage 1 candidate extraction: kept={stage1_kept}, filtered={stage1_filtered}"
        )
        self.logger.info(
            f"{prefix} - Stage 2 unique-candidate constraint: kept={stage2_kept}, filtered={stage2_filtered}"
        )
        self.logger.info(
            f"{prefix} - Stage 3 content quality filter: kept={final_kept}, filtered={stage3_filtered}"
        )
        self.logger.info(
            f"{prefix} - Stage 3 detail (post-fetch content filter): kept={stage3_content_kept}, filtered={stage3_content_filtered}"
        )
        self.logger.info(
            f"{prefix} - Stage 3 detail (post-fetch content reasons): "
            f"pr_filtered={summary.get('issue_filter_github_pr', 0)}, "
            f"empty_body_filtered={summary.get('issue_filter_body_empty', 0)}, "
            f"url_body_filtered={summary.get('issue_filter_body_url', 0)}, "
            f"title_duplicate_filtered={summary.get('stage3_basic_title_filtered', 0)}, "
            f"other_invalid_issue_filtered={summary.get('stage3_no_valid_issue', 0)}"
        )

    @staticmethod
    def derive_stage_audit_output_paths(output_file: str) -> Dict[str, str]:
        path = Path(output_file)
        return {
            'stage1_no_candidate': str(path.with_name(f"{path.stem}_stage1_no_candidate{path.suffix}")),
            'stage2_multi_candidate_only': str(path.with_name(f"{path.stem}_stage2_multi_candidate_only{path.suffix}")),
            'stage3_content_filtered': str(path.with_name(f"{path.stem}_stage3_content_filtered{path.suffix}")),
        }

    def switch_to_github_token(self, token_index: int):
        self.current_github_token_index = token_index
        self.github_token = self.github_tokens[token_index] if self.github_tokens else None
        if self.github_token:
            self.github_session.headers['Authorization'] = f'token {self.github_token}'
            self.logger.info(f"Switched to GitHub token {token_index + 1}/{len(self.github_tokens)}")
        else:
            self.github_session.headers.pop('Authorization', None)

    def rotate_github_token(self, reason: str) -> bool:
        with self.github_token_lock:
            if len(self.github_tokens) <= 1:
                return False

            next_index = self.current_github_token_index + 1
            if next_index >= len(self.github_tokens):
                return False

            self.logger.warning(f"Current GitHub token is invalid or rate limited, reason: {reason}; switching to next token")
            self.switch_to_github_token(next_index)
            with self.github_rate_limit_lock:
                self.request_count = 0
                self.reset_time = time.time() + 3600
            return True

    def wait_for_rate_limit(self):
        """Handle GitHub API rate limiting."""
        sleep_time = 0
        with self.github_rate_limit_lock:
            self.request_count += 1

            # Check whether the counter needs a reset
            if time.time() > self.reset_time:
                self.request_count = 1
                self.reset_time = time.time() + 3600  # Reset after 1 hour

            # If close to the limit, wait
            if self.request_count >= 4950:  # Leave some buffer
                sleep_time = self.reset_time - time.time()

        if sleep_time > 0:
            self.logger.warning(f"Approaching rate limit, waiting {sleep_time:.0f} seconds")
            time.sleep(sleep_time)
            with self.github_rate_limit_lock:
                self.request_count = 1
                self.reset_time = time.time() + 3600

        # Base delay to avoid overly fast requests
        time.sleep(0.1)

    @staticmethod
    def parse_retry_after(headers) -> Optional[float]:
        """Parse Retry-After seconds from response headers."""
        retry_after = headers.get('Retry-After') if headers else None
        if retry_after is None:
            return None
        try:
            return max(float(retry_after), 0.0)
        except (TypeError, ValueError):
            return None

    def get_retry_delay(self, attempt: int, response=None, base_delay: float = 1.0) -> float:
        """Prefer Retry-After; otherwise use exponential backoff with small jitter."""
        retry_after = self.parse_retry_after(response.headers if response is not None else None)
        if retry_after is not None:
            return max(retry_after, 1.0)
        return min(base_delay * (2 ** attempt), self.retry_backoff_cap) + random.uniform(0, 0.5)

    def wait_before_retry(self, platform: str, context: str, attempt: int, response=None, base_delay: float = 1.0):
        """Wait before retrying and log a unified message."""
        delay = self.get_retry_delay(attempt=attempt, response=response, base_delay=base_delay)
        status_code = response.status_code if response is not None else "request-exception"
        self.logger.warning(
            f"{platform} request will retry: context={context}, status={status_code}, "
            f"attempt={attempt + 1}/{self.max_retry_attempts}, sleep={delay:.1f}s"
        )
        time.sleep(delay)

    def update_github_rate_limit_state(self, response):
        """Sync GitHub rate-limit window to reduce client/server drift."""
        limit = response.headers.get('X-RateLimit-Limit')
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset_at = response.headers.get('X-RateLimit-Reset')

        try:
            limit_value = int(limit)
            remaining_value = int(remaining)
            reset_value = float(reset_at)
        except (TypeError, ValueError):
            return

        with self.github_rate_limit_lock:
            self.request_count = max(limit_value - remaining_value, 0)
            self.reset_time = reset_value

    def should_retry_jira_response(self, response) -> bool:
        """Retry only for clearly recoverable Jira rate limit/service hiccups."""
        if response.status_code in [429, 500, 502, 503, 504]:
            return True
        if response.status_code != 403:
            return False

        if self.parse_retry_after(response.headers) is not None:
            return True

        response_text = (response.text or '')[:500].lower()
        return 'rate limit' in response_text or 'too many requests' in response_text

    def handle_github_rate_limit(self, response, attempt: int = 0, context: str = "GitHub request") -> bool:
        """Decide whether to wait until rate-limit reset based on GitHub headers."""
        self.update_github_rate_limit_state(response)
        remaining = response.headers.get('X-RateLimit-Remaining')
        reset_at = response.headers.get('X-RateLimit-Reset')

        if remaining == '0' and self.rotate_github_token("rate limit reached"):
            return True

        sleep_time = None
        if remaining == '0' and reset_at:
            try:
                sleep_time = max(int(reset_at) - int(time.time()), 1)
            except ValueError:
                sleep_time = None

        if sleep_time is None:
            retry_after = self.parse_retry_after(response.headers)
            if retry_after is not None:
                sleep_time = max(retry_after, 1.0)

        if sleep_time is None:
            sleep_time = self.get_retry_delay(attempt=attempt, response=response)

        self.logger.warning(
            f"{context} hit GitHub rate limit, status={response.status_code}, retrying after {sleep_time:.1f} seconds"
        )
        time.sleep(sleep_time)
        return True

    def extract_repo_info(self, repo_url: str) -> Tuple[str, str]:
        """Extract owner and repo name from a repository URL."""
        # Handle multiple URL formats
        patterns = [
            r'https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$',
            r'git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$'
        ]

        for pattern in patterns:
            match = re.match(pattern, repo_url.strip())
            if match:
                return match.group(1), match.group(2)

        raise ValueError(f"Unable to parse repo URL: {repo_url}")

    def get_commit_info(self, owner: str, repo: str, sha: str) -> Optional[Dict]:
        """Fetch commit details."""
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
        context = f"Fetch commit {owner}/{repo}/{sha}"

        for attempt in range(self.max_retry_attempts):
            try:
                self.wait_for_rate_limit()
                response = self.github_session.get(url, timeout=self.request_timeout)
                self.update_github_rate_limit_state(response)

                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    self.logger.warning(f"Commit not found: {owner}/{repo}/{sha}")
                    return None
                elif response.status_code == 401:
                    if self.rotate_github_token("401 bad credentials while fetching commit"):
                        continue
                    self.logger.error(f"Failed to fetch commit info: 401 - {response.text}")
                    return None
                elif response.status_code in [403, 429]:
                    if attempt < self.max_retry_attempts - 1 and self.handle_github_rate_limit(
                        response,
                        attempt=attempt,
                        context=context,
                    ):
                        continue
                    self.logger.error(f"Failed to fetch commit info: {response.status_code} - {response.text}")
                    return None
                elif response.status_code in [500, 502, 503, 504]:
                    if attempt < self.max_retry_attempts - 1:
                        self.wait_before_retry("GitHub", context, attempt, response=response)
                        continue
                    self.logger.warning(f"Failed to fetch commit info: {response.status_code} - {response.text}")
                    return None
                else:
                    self.logger.warning(f"Failed to fetch commit info: {response.status_code} - {response.text}")
                    return None

            except requests.RequestException as e:
                if attempt < self.max_retry_attempts - 1:
                    self.wait_before_retry("GitHub", context, attempt)
                    continue
                self.logger.error(f"Request failed: {e}")
                return None

    def extract_jira_issue(self, sha: str, repo: str, message: str) -> List[str]:
        """
            Extract Jira issue keys from a commit message.

            Args:
                sha: commit sha
                message: commit message containing issue references
                repo: repository name

            Returns:
                list: full Jira issue key list, e.g. ["SPARK-123", "HDFS-456"]
            """
        if not message:
            return []

        normalized_repo = (repo or "").strip().lower()
        if not normalized_repo or normalized_repo in NO_JIRA_REPOS:
            return []

        # In full.jsonl cross-repo mode, many Jira keys do not match repo.upper(),
        # so prefer explicit mapping; fall back to repo.upper() when missing.
        jira_keys = REPO_JIRA_KEYS.get(normalized_repo, [normalized_repo.upper()])
        jira_keys = [key.strip().upper() for key in jira_keys if key and key.strip()]
        if not jira_keys:
            return []

        # Match the more explicit [KEY-123] first; if not found, match KEY-123 anywhere.
        key_group = "|".join(re.escape(key) for key in sorted(set(jira_keys), key=len, reverse=True))
        patterns = [
            rf'\[({key_group})-(\d+)\]',
            rf'\b({key_group})-(\d+)\b',
        ]

        extracted_issue_keys = []
        seen_issue_keys = set()
        for pattern in patterns:
            for match in re.finditer(pattern, message, re.IGNORECASE):
                issue_key = f"{match.group(1).upper()}-{match.group(2)}"
                if issue_key not in seen_issue_keys:
                    extracted_issue_keys.append(issue_key)
                    seen_issue_keys.add(issue_key)
            if extracted_issue_keys:
                break

        return extracted_issue_keys

    def extract_github_issue(self, sha: str, message: str) -> List[str]:
        """Extract GitHub issue references from a commit message."""
        # Prefer formats like "Closes #9931 from"
        priority_pattern = r'(?:Closes|Fixes|Resolves|Close|Fix|Resolve)\s*#(\d+)\s*(?:from|in|$)'
        priority_match = re.search(priority_pattern, message, re.IGNORECASE)

        if priority_match:
            return [priority_match.group(1)]

        # If no priority format matches, fall back to the original patterns
        patterns = [
            r'#(\d+)',
            r'gh-(\d+)',
            r'issue[#\s]*(\d+)',
            r'fixes?[#\s]*(\d+)',
            r'closes?[#\s]*(\d+)',
            r'resolves?[#\s]*(\d+)'
        ]

        issues = set()
        for pattern in patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            issues.update(matches)

        return list(issues)

    def normalize_commit_subject_for_title_compare(self, commit_message: str) -> str:
        """Take the first line of commit message and drop leading [SPARK-xxx][SQL]-style prefixes."""
        if not commit_message:
            return ""

        subject = commit_message.strip().splitlines()[0].strip()
        return re.sub(r'^\s*(?:\[[^\]]+\]\s*)+', '', subject).strip()

    def normalize_text_for_basic_title_compare(self, text: str) -> str:
        """Basic title compare: normalize case, punctuation, and whitespace without removing stopwords."""
        text = (text or '').strip().lower()
        if not text:
            return ""

        text = re.sub(r'[\W_]+', ' ', text, flags=re.UNICODE)
        return re.sub(r'\s+', ' ', text).strip()

    def should_skip_issue_by_basic_title_match(self, commit_message: str, issue_title: str) -> bool:
        """Basic filter: after prefix removal, skip if titles match in case/punctuation/whitespace."""
        normalized_commit_subject = self.normalize_commit_subject_for_title_compare(commit_message)
        return bool(
            normalized_commit_subject
            and issue_title
            and self.normalize_text_for_basic_title_compare(normalized_commit_subject)
            == self.normalize_text_for_basic_title_compare(issue_title)
        )

    def get_issue_info_from_github(self, owner: str, repo: str, issue_number: str) -> Optional[Dict]:
        """Fetch GitHub issue details."""
        url = f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}"
        context = f"Fetch GitHub issue {owner}/{repo}#{issue_number}"
        self.set_last_issue_reject_reason(None)

        for attempt in range(self.max_retry_attempts):
            try:
                self.wait_for_rate_limit()
                response = self.github_session.get(url, timeout=self.request_timeout)
                self.update_github_rate_limit_state(response)

                if response.status_code == 200:
                    issue_data = response.json()
                    break
                elif response.status_code == 404:
                    self.set_last_issue_reject_reason('github_issue_not_found')
                    self.logger.error(f"GitHub issue not found: {owner}/{repo}#{issue_number}")
                    return None
                elif response.status_code == 410:  # Gone - issue may be deleted
                    self.set_last_issue_reject_reason('github_issue_gone')
                    self.logger.error(f"GitHub issue unavailable: {owner}/{repo}#{issue_number}")
                    return None
                elif response.status_code == 401:
                    if self.rotate_github_token(f"401 bad credentials while fetching github issue #{issue_number}"):
                        continue
                    self.set_last_issue_reject_reason('github_issue_unauthorized')
                    self.logger.error(f"Failed to fetch GitHub issue: 401 - {response.text}")
                    return None
                elif response.status_code in [403, 429]:
                    if attempt < self.max_retry_attempts - 1 and self.handle_github_rate_limit(
                        response,
                        attempt=attempt,
                        context=context,
                    ):
                        continue
                    self.set_last_issue_reject_reason('github_issue_rate_limited')
                    self.logger.error(f"Failed to fetch GitHub issue: {response.status_code} - {response.text}")
                    return None
                elif response.status_code in [500, 502, 503, 504]:
                    if attempt < self.max_retry_attempts - 1:
                        self.wait_before_retry("GitHub", context, attempt, response=response)
                        continue
                    self.set_last_issue_reject_reason('github_issue_server_error')
                    self.logger.error(f"Failed to fetch GitHub issue: {response.status_code}")
                    return None
                else:
                    self.set_last_issue_reject_reason('github_issue_request_failed')
                    self.logger.error(f"Failed to fetch GitHub issue: {response.status_code}")
                    return None
            except requests.RequestException as e:
                if attempt < self.max_retry_attempts - 1:
                    self.wait_before_retry("GitHub", context, attempt)
                    continue
                self.set_last_issue_reject_reason('github_issue_request_exception')
                self.logger.error(f"Request failed: {e}")
                return None
        else:
            return None

        # Extract issue fields.
        # GitHub /issues/{number} returns both issues and pull requests.
        # If the response includes pull_request, it is a PR and should be skipped.
        if issue_data.get('pull_request'):
            self.increment_quality_stat('issue_filter_github_pr')
            self.set_last_issue_reject_reason('github_pr')
            self.logger.debug(f"Github #{issue_number} is a pull request, skip: {owner}/{repo}#{issue_number}")
            return None

        issue_body = issue_data['body'] or ''

        # Conditions: not empty and not a pure URL
        if not issue_body.strip():
            self.increment_quality_stat('issue_filter_body_empty')
            self.set_last_issue_reject_reason('issue_body_empty')
            self.logger.debug(f"owner {owner}, repo {repo}, Issue {issue_data['number']} body is empty")
            return None
        elif re.match(r'^https?://[^\s]+$', issue_body.strip()):
            self.increment_quality_stat('issue_filter_body_url')
            self.set_last_issue_reject_reason('issue_body_url')
            self.logger.debug(f"Issue {issue_data['number']} body is a URL: {issue_body.strip()}")
            return None

        return {
            'issue_number': issue_data['number'],
            'issue_title': issue_data['title'],
            'issue_state': issue_data['state'],
            'issue_created_at': issue_data['created_at'],
            'issue_closed_at': issue_data.get('closed_at'),
            'issue_user': issue_data['user']['login'],
            'issue_labels': [label['name'] for label in issue_data.get('labels', [])],
            'issue_body': issue_body
        }

    def get_issue_info_from_jira(self, owner: str, repo: str, issue_number: str) -> Optional[Dict]:
        """Fetch Jira issue details."""
        # extract_jira_issue now returns full issue keys (e.g., SPARK-123).
        # Backward compatibility: if a pure number is passed, fall back to repo.upper() prefix.
        self.set_last_issue_reject_reason(None)
        if re.fullmatch(r'[A-Z][A-Z0-9_-]*-\d+', issue_number or '', re.IGNORECASE):
            issue_key = issue_number.upper()
        else:
            issue_key = f"{repo.upper()}-{issue_number}"

        url = f"https://issues.apache.org/jira/rest/api/2/issue/{issue_key}"
        # self.logger.info(url)
        context = f"Fetch Jira issue {issue_key}"

        for attempt in range(self.max_retry_attempts):
            try:
                response = self.jira_session.get(url, timeout=self.request_timeout)
            except requests.RequestException as e:
                if attempt < self.max_retry_attempts - 1:
                    self.wait_before_retry("Jira", context, attempt)
                    continue
                self.set_last_issue_reject_reason('jira_request_exception')
                self.logger.error(f"Request failed: {e}")
                return None

            if response.status_code == 200:
                break
            elif response.status_code == 404:
                self.set_last_issue_reject_reason('jira_issue_not_found')
                self.logger.error(f"Jira issue not found: {issue_key}")
                return None
            elif response.status_code == 410:
                self.set_last_issue_reject_reason('jira_issue_gone')
                self.logger.error(f"Jira issue unavailable: {issue_key}")
                return None
            elif response.status_code == 401:
                self.set_last_issue_reject_reason('jira_issue_unauthorized')
                self.logger.error(f"Jira issue unauthorized: {response.status_code} {issue_key}")
                return None
            elif self.should_retry_jira_response(response):
                if attempt < self.max_retry_attempts - 1:
                    self.wait_before_retry("Jira", context, attempt, response=response)
                    continue
                self.set_last_issue_reject_reason('jira_issue_rate_limited_or_server_error')
                self.logger.error(f"Jira issue rate limited or server error: {response.status_code} {issue_key}")
                return None
            else:
                self.set_last_issue_reject_reason('jira_issue_request_failed')
                self.logger.error(f"Failed to fetch Jira issue: {response.status_code} {response}")
                return None
        else:
            return None

        try:
            issue_data = response.json()
        except ValueError as e:
            self.set_last_issue_reject_reason('jira_issue_json_invalid')
            self.logger.error(f"Failed to parse JSON: {e}")
            return None

        # Extract issue fields
        try:
            issue_fields = issue_data['fields']
            issue_body = issue_fields.get('description') or ''

            # Conditions: not empty and not a pure URL
            body_stripped = issue_body.strip()
            if not body_stripped:
                self.increment_quality_stat('issue_filter_body_empty')
                self.set_last_issue_reject_reason('issue_body_empty')
                self.logger.debug(f"owner:{owner}, repo:{repo}, Issue:{issue_key} body is empty")
                return None
            elif re.match(r'^https?://\S+$', body_stripped) and len(body_stripped) < 500:
                self.increment_quality_stat('issue_filter_body_url')
                self.set_last_issue_reject_reason('issue_body_url')
                self.logger.debug(f"Issue {issue_key} body is a URL: {body_stripped}")
                return None

            return {
                'issue_number': issue_data.get('key', issue_key),
                'issue_title': issue_fields['summary'],
                'issue_state': issue_fields['status']['name'],
                'issue_created_at': issue_fields['created'],
                'issue_closed_at': issue_fields.get('updated'),
                'issue_user': issue_fields.get('reporter').get('name'),
                'issue_labels': issue_fields.get('labels', []),
                'issue_body': issue_body
            }

        except KeyError as e:
            self.set_last_issue_reject_reason('jira_issue_missing_field')
            self.logger.error(f"Jira response missing field {e}: {issue_key}")
            return None

    def process_commit(self, commit_sha: str, repo_url: str, commit_info: Dict[str, str] = None) -> Optional[Dict]:
        """Process a single commit and extract related issue info."""
        try:
            self.increment_quality_stat('commits_total')
            # Extract repository info
            owner, repo = self.extract_repo_info(repo_url)

            # Prefer commit info from source data to avoid extra GitHub API calls.
            commit_message = (commit_info or {}).get('commit_message')
            commit_author = (commit_info or {}).get('commit_author')
            commit_date = (commit_info or {}).get('commit_date')
            if not commit_message:
                commit_data = self.get_commit_info(owner, repo, commit_sha)
                if not commit_data:
                    self.increment_quality_stat('source_commit_unavailable')
                    return None
                commit_message = commit_data['commit']['message']
                commit_author = commit_data['commit']['author']['name'] if commit_data['commit']['author'] else None
                commit_date = commit_data['commit']['author']['date'] if commit_data['commit']['author'] else None

            # Try the preferred platform in config first; if it returns PRs, empty bodies,
            # or other invalid issues, fall back to the other platform.
            preferred_platform = self.config.get(repo_url, self.default_preferred_platform)
            if preferred_platform == GITHUB:
                platform_order = [GITHUB, JIRA]
            elif preferred_platform == JIRA:
                platform_order = [JIRA, GITHUB]
            else:
                platform_order = [GITHUB, JIRA]

            issue_number = None
            issue_info = None
            issue_source_platform = None
            had_any_candidate = False
            had_single_candidate = False
            filtered_by_basic_title = False
            multi_candidate_events = []
            candidate_attempts = []
            commit_subject = self.normalize_commit_subject_for_title_compare(commit_message)

            for current_platform in platform_order:
                if current_platform == GITHUB:
                    issue_refs = self.extract_github_issue(sha=commit_sha, message=commit_message)
                else:
                    issue_refs = self.extract_jira_issue(sha=commit_sha, repo=repo, message=commit_message)

                if len(issue_refs) == 0:
                    continue

                had_any_candidate = True
                if len(issue_refs) > 1:
                    multi_candidate_events.append({
                        'platform': current_platform,
                        'issue_refs': issue_refs,
                    })
                    self.logger.debug(
                        f"{current_platform} issue ref count: {len(issue_refs)} repo: {repo} commit_sha: {commit_sha}\n message: {commit_message}"
                    )
                    continue

                had_single_candidate = True
                candidate_issue_number = issue_refs[0]
                if current_platform == GITHUB:
                    candidate_issue_info = self.get_issue_info_from_github(
                        owner=owner, repo=repo, issue_number=candidate_issue_number
                    )
                else:
                    candidate_issue_info = self.get_issue_info_from_jira(
                        owner=owner, repo=repo, issue_number=candidate_issue_number
                    )

                if candidate_issue_info is not None and self.should_skip_issue_by_basic_title_match(
                    commit_message, candidate_issue_info.get('issue_title', '')
                ):
                    filtered_by_basic_title = True
                    candidate_attempts.append({
                        'platform': current_platform,
                        'issue_ref': candidate_issue_number,
                        'reject_reason': 'basic_title_match',
                        'issue_title': candidate_issue_info.get('issue_title', ''),
                    })
                    self.logger.debug(
                        f"skip {current_platform} issue because title matches commit subject after basic normalization: "
                        f"repo: {repo}, commit_sha: {commit_sha}, issue_ref: {candidate_issue_number}"
                    )
                    continue

                if candidate_issue_info is not None:
                    issue_number = candidate_issue_number
                    issue_info = candidate_issue_info
                    issue_source_platform = current_platform
                    break

                reject_reason, reject_details = self.consume_last_issue_reject_reason()
                candidate_attempts.append({
                    'platform': current_platform,
                    'issue_ref': candidate_issue_number,
                    'reject_reason': reject_reason or 'issue_fetch_returned_none',
                    **reject_details,
                })
                self.logger.debug(
                    f"{current_platform} issue has no valid result, trying another platform: repo: {repo}, commit_sha: {commit_sha}, issue_ref: {candidate_issue_number}"
                )

            if had_any_candidate:
                self.increment_quality_stat('stage1_any_candidate')
            else:
                self.increment_quality_stat('stage1_no_candidate')
                self.record_stage_audit(
                    'stage1_no_candidate',
                    {
                        'commit_sha': commit_sha,
                        'repo_owner': owner,
                        'repo_name': repo,
                        'repo_url': repo_url,
                        'commit_subject': commit_subject,
                    }
                )

            if had_single_candidate:
                self.increment_quality_stat('stage2_unique_candidate')
            elif had_any_candidate:
                self.increment_quality_stat('stage2_multi_candidate_only')
                self.record_stage_audit(
                    'stage2_multi_candidate_only',
                    {
                        'commit_sha': commit_sha,
                        'repo_owner': owner,
                        'repo_name': repo,
                        'repo_url': repo_url,
                        'commit_subject': commit_subject,
                        'multi_candidate_events': multi_candidate_events,
                    }
                )

            if issue_info is None:
                if had_single_candidate:
                    if filtered_by_basic_title:
                        self.increment_quality_stat('stage3_basic_title_filtered')
                    else:
                        self.increment_quality_stat('stage3_no_valid_issue')
                    self.record_stage_audit(
                        'stage3_content_filtered',
                        {
                            'commit_sha': commit_sha,
                            'repo_owner': owner,
                            'repo_name': repo,
                            'repo_url': repo_url,
                            'commit_subject': commit_subject,
                            'candidate_attempts': candidate_attempts,
                        }
                    )
                self.logger.debug(f"no valid issue found repo: {repo} commit_sha: {commit_sha}\n message: {commit_message}")
                return None

            self.increment_quality_stat('stage3_content_kept')
            return {
                'commit_sha': commit_sha,
                'repo_owner': owner,
                'repo_name': repo,
                'repo_url': repo_url,
                'commit_message': commit_message,
                'commit_author': commit_author,
                'commit_date': commit_date,
                'issue_reference': issue_number,
                'issue_source_platform': issue_source_platform,
                'issue': issue_info,
            }

        except Exception as e:
            self.logger.error(f"Failed to process commit {commit_sha}: {e}")
            return {
                'commit_sha': commit_sha,
                'repo_url': repo_url,
                'error': str(e)
            }

    def crawl_commits(
        self,
        commits_data: List[Dict],
        output_file: str = 'commit_issues.json',
        max_workers: int = 1,
        enable_quality_summary: bool = True,
    ):
        """Crawl issue info for all commits."""
        self.quality_stats_enabled = enable_quality_summary
        if enable_quality_summary:
            self.reset_quality_stats()
            self.reset_stage_audit_records()
        else:
            with self.quality_stats_lock:
                self.last_quality_stats_snapshot = self.create_quality_stats()
        results = []
        max_workers = max(1, int(max_workers or 1))
        # Write progress files into the output_file directory to avoid CWD scattering
        _progress_dir = os.path.dirname(os.path.abspath(output_file))
        os.makedirs(_progress_dir, exist_ok=True)
        self.logger.info(f"Start crawling commit issues, total: {len(commits_data)}, max_workers: {max_workers}")

        if max_workers == 1:
            # Progress bar
            pbar = tqdm(commits_data, desc="Crawling commit info")

            for commit_info in pbar:
                if isinstance(commit_info, dict):
                    commit_sha = commit_info.get('commit_sha')
                    repo_url = commit_info.get('repo_url')
                else:
                    # Tuple format
                    commit_sha, repo_url = commit_info

                if not commit_sha or not repo_url:
                    self.logger.warning(f"Skip invalid data: {commit_info}")
                    continue

                pbar.set_description(f"Processing {commit_sha[:8]}... collected {len(results)} valid records")

                current_commit_info = commit_info if isinstance(commit_info, dict) else None
                result = self.process_commit(commit_sha, repo_url, current_commit_info)
                if result is not None:
                    if result.get('error'):
                        raise RuntimeError(f"Commit processing failed, stopping: {result['commit_sha']} -> {result['error']}")
                    results.append(result)

                self.operated_count += 1

                # Save progress every 500 valid results
                if len(results) > 0 and len(results) % 500 == 0:
                    self.save_progress(results, os.path.join(_progress_dir, f"progress_{self.operated_count}.json"))
                    self.logger.info(f"Processed {self.operated_count} commits")
        else:
            valid_commits = []
            for commit_info in commits_data:
                if isinstance(commit_info, dict):
                    commit_sha = commit_info.get('commit_sha')
                    repo_url = commit_info.get('repo_url')
                else:
                    commit_sha, repo_url = commit_info
                if not commit_sha or not repo_url:
                    self.logger.warning(f"Skip invalid data: {commit_info}")
                    continue
                valid_commits.append(commit_info)

            pbar = tqdm(total=len(valid_commits), desc="Parallel crawl commit info")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_commit = {}
                for commit_info in valid_commits:
                    commit_sha = commit_info.get('commit_sha')
                    repo_url = commit_info.get('repo_url')
                    future = executor.submit(self.process_commit, commit_sha, repo_url, commit_info)
                    future_to_commit[future] = commit_sha

                for future in concurrent.futures.as_completed(future_to_commit):
                    commit_sha = future_to_commit[future]
                    pbar.set_description(f"Processing {commit_sha[:8]}... collected {len(results)} valid records")
                    result = future.result()
                    if result is not None:
                        if result.get('error'):
                            executor.shutdown(wait=False, cancel_futures=True)
                            raise RuntimeError(f"Commit processing failed, stopping: {result['commit_sha']} -> {result['error']}")
                        results.append(result)

                    self.operated_count += 1
                    pbar.update(1)

                    if len(results) > 0 and len(results) % 500 == 0:
                        self.save_progress(results, os.path.join(_progress_dir, f"progress_{self.operated_count}.json"))
                        self.logger.info(f"Processed {self.operated_count} commits")

            pbar.close()

        # Save final results
        self.save_progress(results, os.path.join(_progress_dir, f"progress_{self.operated_count}.json"))
        self.save_progress(results, output_file)
        if enable_quality_summary:
            summary = self.snapshot_quality_stats()
            summary['stage3_content_kept'] = len(results)
            summary['final_kept'] = len(results)
            with self.quality_stats_lock:
                self.last_quality_stats_snapshot = dict(summary)
            self.log_quality_summary(summary, prefix=f"Quality summary [{output_file}]")
            stage_audit_paths = self.derive_stage_audit_output_paths(output_file)
            stage_audit_records = self.snapshot_stage_audit_records()
            for stage_key, file_path in stage_audit_paths.items():
                self.save_progress(stage_audit_records.get(stage_key, []), file_path)
            self.logger.info(
                f"Stage audit files saved: "
                f"stage1={stage_audit_paths['stage1_no_candidate']}, "
                f"stage2={stage_audit_paths['stage2_multi_candidate_only']}, "
                f"stage3={stage_audit_paths['stage3_content_filtered']}"
            )
            self.quality_stats_enabled = False
        self.logger.info(
            f"Processed {self.operated_count} commits, final={len(results)}"
        )
        return results

    def save_progress(self, results: List[Dict], filename: str):
        """Save progress."""
        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def load_commits_from_json_file(self, file_path: str, filter_file_path: str = None, git_url: str = None):
        """Load commit data from a JSONL file; when git_url is None, do not filter by repo."""
        filter_data = set()
        if filter_file_path:
            try:
                with open(filter_file_path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                if git_url is not None:
                    filter_data = {d.get('commit_sha') for d in data if d.get('repo_url') == git_url}
                else:
                    filter_data = {d.get('commit_sha') for d in data}
            except FileNotFoundError:
                self.logger.error(f"Filter file not found: {filter_file_path}")
            except json.JSONDecodeError:
                self.logger.error(f"Filter file is invalid JSON: {filter_file_path}")
            except Exception as e:
                self.logger.error(f"Failed to read filter commit file: {e}")

        commits_data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if (git_url is None or data.get('git_url') == git_url) and data.get('commit_sha') not in filter_data:
                            commits_data.append({
                                'commit_sha': data['commit_sha'],
                                'repo_url': data['git_url'],
                                'commit_message': data.get('message'),
                                'commit_author': data.get('author'),
                                'commit_date': data.get('date'),
                            })
                    except (json.JSONDecodeError, KeyError) as e:
                        self.logger.warning(f"Failed to parse line: {e}, data: {line}")
                        continue
        except FileNotFoundError:
            self.logger.error(f"Commit file not found: {file_path}")
        except json.JSONDecodeError:
            self.logger.error(f"Commit file is invalid JSON: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to read commit file: {e}")
        self.logger.info(f"Commit file loaded, total {len(commits_data)} records")

        return commits_data

    @staticmethod
    def normalize_git_url_filter(git_url: str = None) -> Optional[str]:
        """Treat --git-url all/*/all-repos as cross-repo mode."""
        if git_url is None:
            return None
        normalized = git_url.strip()
        if not normalized:
            return None
        if normalized.lower() in {"all", "*", "all-repos"}:
            return None
        return normalized

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch GitHub/Jira issues for commits")
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

    parser.add_argument(
        "--input-file",
        default=os.path.join(_PROJECT_ROOT, "ApacheCM", "10000_sorted_by_git_url.jsonl"),
        help="Input JSONL file path. Defaults to ApacheCM/10000_sorted_by_git_url.jsonl.",
    )
    parser.add_argument(
        "--output-file",
        default=os.path.join(_SCRIPT_DIR, "data", "spark_issues.json"),
        help="Output file path for fetched results.",
    )
    parser.add_argument(
        "--filter-file",
        default=None,
        help="Optional. Filter file for processed commits; skips commit_sha already in the file.",
    )
    parser.add_argument(
        "--git-url",
        default=SPARKURL,
        help="Repo filter for commits mode; default apache/spark. Use all / * / all-repos for cross-repo mode.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent worker count. 1 for serial; >1 for multithreading.",
    )
    parser.add_argument(
        "--github-token",
        action="append",
        dest="github_tokens",
        default=None,
        help="GitHub tokens (repeatable) to rotate on rate limits; defaults to in-code list if omitted.",
    )
    parser.add_argument(
        "--preferred-platform",
        choices=[GITHUB, JIRA],
        default=JIRA,
        help="Preferred platform order. Default Jira first, then GitHub on failure.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level, default INFO; use DEBUG for detailed filtering traces.",
    )
    args = parser.parse_args()

    # TODO: fill in your own
    github_tokens = args.github_tokens or [
        '',
        '',
    ]

    crawler = ApacheCommitIssueCrawler(
        github_tokens=github_tokens,
        preferred_platform=args.preferred_platform,
        log_level=args.log_level,
    )

    git_url_filter = crawler.normalize_git_url_filter(args.git_url)
    commits = crawler.load_commits_from_json_file(
        file_path=args.input_file,
        filter_file_path=args.filter_file,
        git_url=git_url_filter,
    )

    print('full_commits', len(commits))
    results = crawler.crawl_commits(
        commits,
        output_file=args.output_file,
        max_workers=args.max_workers,
        enable_quality_summary=True,
    )
    print('full_results', len(results))
    print('output', args.output_file)

    # try:
    #     # Initialize crawler
    #     # TODO: fill in your own
    #     crawler = ApacheCommitIssueCrawler(
    #         github_token='',
    #         jira_cookie="""""")
    #
    #     # Test crawler configuration
    #     test = crawler.get_issue_info_from_github(owner='apache', repo='spark', issue_number='42387')
    #     if test is None:
    #         exit(1)
    #     test = crawler.get_issue_info_from_jira(owner='apache', repo='CAMEL', issue_number='13663')
    #     if test is None:
    #         exit(1)
    #
    #     commits_data = crawler.load_commits_from_json_file(file_path="./10000_sorted_by_git_url.jsonl", filter_file_path="./data/spark_issues.json", git_url=SPARKURL)
    #
    #     results = crawler.crawl_commits(commits_data)
    #
    #     # Summary
    #     commits_with_issues = len(results)
    #     total_issues = sum(1 for r in results if r.get('issue'))
    #
    #     print(f"\nCrawl completed!")
    #     print(f"Total commits: {len(commits_data)}")
    #     print(f"Commits with issues: {commits_with_issues}")
    #     print(f"Total issues found: {total_issues}")
    #
    # except Exception as e:
    #     print(f"Execution failed: {e}")
