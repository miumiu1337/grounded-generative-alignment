"""Bounded HTTP range reads for small, pinned FineWiki article lookups."""

from collections import OrderedDict
import io
import math
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests


class RangeReader(io.RawIOBase):
    block_size = 1024 * 1024

    def __init__(self, url, deadline_seconds=240, block_size=None):
        super().__init__()
        self.session = None
        if type(deadline_seconds) not in (int, float) or not math.isfinite(deadline_seconds) or deadline_seconds <= 0:
            raise ValueError("FineWiki lookup deadline must be finite and positive")
        if block_size is not None:
            if type(block_size) is not int or not 65536 <= block_size <= 8 * 1024 * 1024:
                raise ValueError("FineWiki range block size must be 64 KiB through 8 MiB")
            self.block_size = block_size
        self.url, self.position = url, 0
        self.deadline = time.monotonic() + deadline_seconds
        self.session = requests.Session()
        self.cache = OrderedDict()
        try:
            self._range(0, 0)
        except BaseException:
            self.session.close()
            raise

    @staticmethod
    def _retry_delay(response, attempt):
        delay = 2 ** attempt
        hint = response.headers.get("Retry-After")
        if hint:
            try:
                seconds = float(hint)
            except ValueError:
                try:
                    parsed = parsedate_to_datetime(hint)
                    seconds = (parsed - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    seconds = 0
            if math.isfinite(seconds):
                delay = max(delay, seconds)
        elif response.status_code == 429:
            match = re.search(r"(?:^|;)\s*t=(\d+)", response.headers.get("RateLimit", ""))
            if match:
                delay = max(delay, int(match.group(1)))
        return delay

    def _request(self, start, end):
        for attempt in range(3):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("FineWiki Parquet range lookup exceeded its deadline")
            response = self.session.get(self.url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                timeout=(min(10, remaining), min(45, remaining)), stream=True)
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
            delay = self._retry_delay(response, attempt)
            status = response.status_code
            response.close()
            if attempt == 2:
                raise RuntimeError(f"FineWiki range server remains unavailable (HTTP {status}); retry after {delay:g}s; lookup checkpoints are preserved")
            if delay >= self.deadline - time.monotonic():
                raise TimeoutError(f"FineWiki range server requires a {delay:g}s retry wait beyond the remaining lookup deadline")
            while delay > 0:
                chunk = min(30, delay)
                time.sleep(chunk)
                delay -= chunk
        raise RuntimeError("FineWiki range retry budget exhausted")

    def _range(self, start, end):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("FineWiki Parquet range lookup exceeded its deadline")
        with self._request(start, end) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"FineWiki range server returned HTTP {response.status_code}")
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
            if response.status_code != 206 or not match:
                raise ValueError("FineWiki server did not honor a bounded byte range")
            actual_start, actual_end, size = map(int, match.groups())
            if actual_start != start or actual_end != min(end, size - 1) or (hasattr(self, "size") and size != self.size):
                raise ValueError("FineWiki Parquet byte-range identity changed")
            self.size = size
            chunks, length = [], 0
            for chunk in response.iter_content(65536):
                length += len(chunk)
                if length > actual_end - actual_start + 1:
                    raise ValueError("FineWiki range response exceeded its declared length")
                if time.monotonic() >= self.deadline:
                    raise TimeoutError("FineWiki Parquet lookup exceeded its deadline while receiving bytes")
                chunks.append(chunk)
            if length != actual_end - actual_start + 1:
                raise ValueError("Incomplete FineWiki Parquet range response")
            return b"".join(chunks)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        position = offset + (0 if whence == io.SEEK_SET else self.position if whence == io.SEEK_CUR else self.size if whence == io.SEEK_END else -1)
        if whence not in (io.SEEK_SET, io.SEEK_CUR, io.SEEK_END) or position < 0:
            raise ValueError("Invalid Parquet seek")
        self.position = position
        return position

    def read(self, size=-1):
        if size is None or size < 0:
            size = self.size - self.position
        end = min(self.size, self.position + size)
        chunks = []
        while self.position < end:
            block = self.position // self.block_size
            if block not in self.cache:
                start = block * self.block_size
                self.cache[block] = self._range(start, min(self.size - 1, start + self.block_size - 1))
                if len(self.cache) > 8:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(block)
            count = min(end - self.position, self.block_size - self.position % self.block_size)
            chunks.append(self.cache[block][self.position % self.block_size:self.position % self.block_size + count])
            self.position += count
        return b"".join(chunks)

    def close(self):
        if self.session is not None:
            self.session.close()
        super().close()


def selected_rows(url, page_ids, *, filter_column="page_id", block_size=None, deadline_seconds=240):
    import pyarrow.parquet as parquet

    wanted, rows = set(page_ids), []
    if filter_column not in {"page_id", "wikidata_id"}:
        raise ValueError("Only article page IDs or Wikidata topic IDs may select FineWiki rows")
    valid_ids = (all(type(value) is int and value > 0 for value in wanted) if filter_column == "page_id" else
                 all(isinstance(value, str) and re.fullmatch(r"Q[1-9][0-9]*", value) for value in wanted))
    if not wanted or not valid_ids:
        raise ValueError("Invalid FineWiki article/topic lookup identifiers")
    columns = ["id", "page_id", "title", "wikidata_id", "version", "text", "url", "in_language",
               "date_modified", "wikiname", "has_math"]
    with RangeReader(url, block_size=block_size, deadline_seconds=deadline_seconds) as reader:
        file = parquet.ParquetFile(reader, pre_buffer=False)
        for group in range(file.num_row_groups):
            ids = file.read_row_group(group, columns=[filter_column], use_threads=False).column(0).to_pylist()
            indices = [index for index, page_id in enumerate(ids) if page_id in wanted]
            if indices:
                table = file.read_row_group(group, columns=columns, use_threads=False)
                rows.extend(table.take(indices).to_pylist())
    return rows
