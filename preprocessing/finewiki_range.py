"""Bounded HTTP range reads for small, pinned FineWiki article lookups."""

from collections import OrderedDict
import io
import re
import time

import requests


class RangeReader(io.RawIOBase):
    block_size = 1024 * 1024

    def __init__(self, url, deadline_seconds=240):
        super().__init__()
        self.url, self.position = url, 0
        self.deadline = time.monotonic() + deadline_seconds
        self.session = requests.Session()
        self.cache = OrderedDict()
        try:
            self._range(0, 0)
        except BaseException:
            self.session.close()
            raise

    def _range(self, start, end):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("FineWiki Parquet range lookup exceeded its deadline")
        with self.session.get(self.url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                              timeout=(min(10, remaining), min(45, remaining)), stream=True) as response:
            response.raise_for_status()
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
        self.session.close()
        super().close()


def selected_rows(url, page_ids):
    import pyarrow.parquet as parquet

    wanted, rows = set(page_ids), []
    columns = ["id", "page_id", "title", "wikidata_id", "version", "text", "url", "in_language",
               "date_modified", "wikiname", "has_math"]
    with RangeReader(url) as reader:
        file = parquet.ParquetFile(reader, pre_buffer=False)
        for group in range(file.num_row_groups):
            ids = file.read_row_group(group, columns=["page_id"], use_threads=False).column(0).to_pylist()
            indices = [index for index, page_id in enumerate(ids) if page_id in wanted]
            if indices:
                table = file.read_row_group(group, columns=columns, use_threads=False)
                rows.extend(table.take(indices).to_pylist())
    return rows
