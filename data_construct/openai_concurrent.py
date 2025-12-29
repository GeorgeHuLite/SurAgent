import time
from collections import deque
import asyncio
from openai import AsyncOpenAI
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Callable
import openai
from openai.types.chat import ChatCompletion
import logging
from datetime import datetime
from pathlib import Path
import json
import gzip


@dataclass
class RequestResult:
    """请求结果记录数据类"""

    request_id: str
    success: bool
    response: Dict = None
    request_data: Dict = None
    error: str = None
    tokens_used: int = 0
    timestamp: str = ""


class RateLimiter:
    """本地估算的请求速率限制器
    """
    def __init__(self, max_requests_per_minute: int = 100, max_tokens_per_minute: int = 1000):
        self.rpm = max_requests_per_minute
        self.tpm = max_tokens_per_minute
        self.request_times = deque()
        self.token_usage = deque()
        # 用于速率限制器自身的锁
        self.lock = asyncio.Lock()

    async def can_make_request(self) -> bool:
        """检查是否还能发起调用请求"""
        async with self.lock:
            now = time.time()
            minute_ago = now - 60

            # 清理一分钟外的记录
            while self.request_times and self.request_times[0] < minute_ago:
                self.request_times.popleft()
            while self.token_usage and self.token_usage[0]["timestamp"] < minute_ago:
                self.token_usage.popleft()

            # 计算是否达到限制
            if len(self.request_times) >= self.rpm:
                return False
            total_tokens = sum(record["tokens"] for record in self.token_usage)
            if total_tokens >= self.tpm:
                return False

            return True

    async def record_request(self, token_used: int = 0):
        """记录请求情况"""
        async with self.lock:
            now = time.time()
            self.request_times.append(now)
            if token_used > 0:
                self.token_usage.append({"timestamp": now, "tokens": token_used})

    async def wait_if_needed(self):
        """睡眠等待函数，直到满足RPM和TPM条件，每秒检查一次"""
        while not await self.can_make_request():
            await asyncio.sleep(1)


class OpenAIConcurrentClient:
    """openai库并发调用请求客户端
    """
    def __init__(
        self,
        api_key: str,
        entry_url: str,
        log_archieve_filepath: Path,
        max_retries: int = 3,
        max_concurrent: int = 5,
        max_requests_per_minute: int = 100,
        max_tokens_per_minute: int = 1000,
    ):
        """并发请求客户端

        Args:
            api_key (str): 调用模型需要用到的apikey
            entry_url (str): 调用模型的入口URL
            log_archieve_filepath (Path): 完整归档日志文件的保存路径
            max_retries (int, optional): 最多重试次数，默认为3
            max_concurrent (int, optional): 同时等待API返回的最大请求数量，默认为5
            max_requests_per_minute (int, optional): 指定预估RPM速率限制，建议略小于API的RPM限制
            max_tokens_per_minute (int, optional): 指定预估TPM速率限制，建议小于API的RPM限制减去一次请求要消耗的token数量
        """
        self.client = AsyncOpenAI(api_key=api_key, base_url=entry_url)
        self.rate_limiter = RateLimiter(
            max_requests_per_minute=max_requests_per_minute, max_tokens_per_minute=max_tokens_per_minute
        )
        # 并发数量限制信号量
        self.semaphore = asyncio.Semaphore(max_concurrent)

        self.max_retries = max_retries
        self.result_filepath = log_archieve_filepath

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)

    async def _make_single_request(
        self, completion_request_data: Dict[str, Any], request_id: str, retry_counter: int = 0
    ):
        """用于内部的请求发起，请勿直接调用"""
        try:
            await self.rate_limiter.wait_if_needed()
            await self.rate_limiter.record_request()
            response: ChatCompletion = await self.client.chat.completions.create(**completion_request_data)
            token_used = 0
            if hasattr(response, "usage") and response.usage:
                token_used = response.usage.total_tokens
                await self.rate_limiter.record_request(token_used)
            response_dict = {
                "id": response.id,
                "created": response.created,
                "model": response.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {"role": choice.message.role, "content": choice.message.content},
                        "finish_reason": choice.finish_reason,
                    }
                    for choice in response.choices
                ],
                "usage": (
                    {
                        "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                        "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                        "total_tokens": response.usage.total_tokens if response.usage else 0,
                    }
                    if response.usage
                    else None
                ),
            }

            result = RequestResult(
                request_id=request_id,
                success=True,
                response=response_dict,
                tokens_used=token_used,
                timestamp=datetime.now().isoformat(),
                request_data=completion_request_data,
            )
            self.logger.info(f"请求 {request_id} 成功完成，使用 {token_used} tokens")
            return result
        except openai.RateLimitError as e:
            self.logger.warning(f"请求 {request_id} 受到速率限制: {e}")
            if retry_counter < self.max_retries:
                delay = 1.0 * (2**retry_counter)
                self.logger.info(f"等待 {delay}s 后重试请求 {request_id}")
                await asyncio.sleep(delay)
                return await self._make_single_request(completion_request_data, request_id, retry_counter + 1)
            else:
                return RequestResult(
                    request_id=request_id,
                    success=False,
                    timestamp=datetime.now().isoformat(),
                    error=f"RateLimitError after {self.max_retries} retries: {str(e)}",
                    request_data=completion_request_data,
                )
        except Exception as e:
            self.logger.error(f"请求 {request_id} 未知错误: {e}")
            return RequestResult(
                request_id=request_id,
                success=False,
                timestamp=datetime.now().isoformat(),
                error=f"error: {str(e)}",
                request_data=completion_request_data,
            )

    async def _save_result(self, result: RequestResult):
        """保存请求结果到文件"""
        try:
            with gzip.open(self.result_filepath, "at", encoding="utf-8") as fo:
                json.dump(asdict(result), fo, ensure_ascii=False)
                fo.write("\n")
        except Exception as e:
            self.logger.error(f"结果保存至文件失败: {e}")

    async def _process_request(self, request_item: Dict) -> RequestResult:
        """发起调用请求

        Args:
            request_item (Dict): 至少包含用于具体请求内容的request
        """
        async with self.semaphore:
            request_id = request_item.get("id", str(time.time()))
            request_data = request_item["request"]

            result = await self._make_single_request(request_data, request_id)
            await self._save_result(result)

            return result

    async def batch_requests(
        self, requests: List[Dict[str, Any]], progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[RequestResult]:
        """批量处理请求

        批量处理requests中的所有请求

        Args:
            requests (List[Dict[str, Any]]): 请求列表，每个请求应包含 'id' 和 'request' 字段
            progress_callback (Optional[Callable[[int, int], None]], optional): 进度回调函数，接收 (completed: int, total: int) 参数

        Returns:
            List[RequestResult]: 请求结果列表
        """
        self.logger.info(f"开始批量处理 {len(requests)} 个请求")

        # 创建任务
        tasks = [self._process_request(req) for req in requests]

        # 执行任务并收集结果
        completed = 0
        results = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, len(requests))

            self.logger.info(f"进度: {completed}/{len(requests)}")

        # 统计结果
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        total_tokens = sum(r.tokens_used for r in results)

        self.logger.info(f"批量请求完成: 成功 {successful}, 失败 {failed}, 总tokens {total_tokens}")

        return results

    async def close(self):
        await self.client.close()
