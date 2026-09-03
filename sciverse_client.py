"""
Sciverse API 客户端 (v2.0 增强版)
提供文献检索 (meta-search)、语义检索 (agentic-search)、多chunk内容获取能力
"""

import json
import hashlib
import os
import random
import urllib.request
import urllib.error
import urllib.parse
import time


class SciverseClient:
    """Sciverse 科研文献 API 客户端"""

    def __init__(self, api_key, base_url=None, max_retries=3, cache_dir=None):
        self.base_url = (base_url or os.environ.get("SCIVERSE_BASE_URL") or "https://api.sciverse.space").rstrip("/")
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self.max_retries = max_retries
        self.call_count = 0
        self.cache_hits = 0
        self.request_attempt_count = 0
        self.min_interval_seconds = float(os.environ.get("SCIVERSE_MIN_INTERVAL_SECONDS", "2.0"))
        self._last_request_started = 0.0
        default_cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "sciverse")
        self.cache_dir = os.path.abspath(cache_dir or os.environ.get("SCIVERSE_CACHE_DIR", default_cache))
        self.cache_enabled = True
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
        except OSError:
            # Read-only deployments remain usable; only persistent caching is disabled.
            self.cache_enabled = False

    def meta_search(self, query, page_size=10, year_from=None, collection="papers"):
        """结构化文献检索，返回论文元数据列表"""
        payload = {
            "query": query,
            "collection": collection,
            "fields": [
                "title", "doi", "publication_published_year",
                "publication_venue_name_unified", "citation_count",
                "author",
            ],
            "page": 1,
            "page_size": page_size,
        }
        if year_from:
            payload["filters"] = [
                {"field": "publication_published_year", "operator": "FILTER_OP_GTE", "value": year_from}
            ]

        resp = self._post("/meta-search", payload)
        return resp.get("results", [])

    def agentic_search(self, query, top_k=5):
        """语义检索，返回包含正文 chunk 的命中结果"""
        payload = {
            "query": query,
            "top_k": top_k,
        }
        resp = self._post("/agentic-search", payload)
        return resp.get("hits", [])

    def agentic_search_multi(self, queries, top_k=5):
        """多条查询语义检索，自动去重，返回更丰富的内容块"""
        all_hits = []
        seen_doc_chunk = set()

        for query in queries:
            hits = self.agentic_search(query, top_k=top_k)
            for hit in hits:
                doc_id = hit.get("doc_id", "")
                chunk_id = hit.get("chunk_id", "")
                dedup_key = f"{doc_id}_{chunk_id}"
                if dedup_key not in seen_doc_chunk:
                    seen_doc_chunk.add(dedup_key)
                    all_hits.append(hit)
            time.sleep(0.3)

        return all_hits

    def get_content(self, doc_id, offset=0, limit=4096):
        """按 doc_id 和 offset 回读正文上下文"""
        params = "?" + urllib.parse.urlencode({
            "doc_id": doc_id,
            "offset": offset,
            "limit": limit,
        })
        return self._get(f"/content{params}")

    def get_content_multi(self, doc_id, num_chunks=3, chunk_size=3000):
        """获取同一文档的多个内容块，用于深度知识抽取"""
        all_content = []
        offset = 0
        for _ in range(num_chunks):
            result = self.get_content(doc_id, offset=offset, limit=chunk_size)
            # Current Sciverse schema uses `text`; keep `content` as a
            # compatibility fallback for older deployments.
            text = result.get("text") or result.get("content") or ""
            if not text:
                break
            all_content.append(text)

            next_offset = result.get("next_offset")
            if result.get("more") is False or next_offset is None:
                break
            offset = next_offset
            time.sleep(0.2)
        return all_content

    def get_paper_full_text(self, doc_id, max_chars=8000):
        """获取论文尽可能多的正文文本，用于知识抽取"""
        if not doc_id:
            return ""
        chunks = self.get_content_multi(doc_id, num_chunks=2, chunk_size=2500)
        return "\n\n".join(chunks)[:max_chars]

    def _post(self, path, payload):
        data = json.dumps(payload).encode("utf-8")
        return self._request("POST", path, data=data, timeout=60)

    def _get(self, path):
        return self._request("GET", path, timeout=30)

    def _request(self, method, path, data=None, timeout=30):
        url = self.base_url + path
        cache_path = self._cache_path(method, path, data)
        cached = self._read_cache(cache_path) if self.cache_enabled else None
        if cached is not None:
            self.cache_hits += 1
            return cached
        for attempt in range(self.max_retries + 1):
            self._throttle()
            req = urllib.request.Request(url, data=data, headers=self.headers, method=method)
            try:
                self.request_attempt_count += 1
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    self.call_count += 1
                    result = json.loads(resp.read().decode("utf-8"))
                    if self.cache_enabled:
                        self._write_cache(cache_path, result)
                    return result
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                retryable = e.code in {429, 500, 502, 503, 504}
                if retryable and attempt < self.max_retries:
                    retry_after = e.headers.get("Retry-After")
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else 2 ** attempt
                    time.sleep(delay + random.random() * 0.25)
                    continue
                print(f"  [Sciverse] HTTP {e.code}: {body[:200]}")
                return {}
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt + random.random() * 0.25)
                    continue
                print(f"  [Sciverse] 请求失败: {e}")
                return {}
        return {}

    def _throttle(self):
        remaining = self.min_interval_seconds - (time.monotonic() - self._last_request_started)
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_started = time.monotonic()

    def _cache_path(self, method, path, data):
        digest = hashlib.sha256(method.encode("utf-8") + b"\0" + path.encode("utf-8") + b"\0" + (data or b"")).hexdigest()
        return os.path.join(self.cache_dir, f"{digest}.json")

    @staticmethod
    def _read_cache(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _write_cache(path, result):
        tmp_path = f"{path}.tmp-{os.getpid()}"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False)
            os.replace(tmp_path, path)
        except OSError:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
