"""Groq API key rotation with rate-limit-aware failover.

Keys come from env var GROQ_API_KEYS=key1,key2,key3 (comma separated).
Use: get_pool().call(lambda client: client.chat.completions.create(...))
"""

import os
import random
import threading
import time

from groq import APIConnectionError, APIStatusError, AuthenticationError, Groq, RateLimitError


class DailyLimitExhausted(BaseException):
    """Every key hit its daily limit. BaseException so broad `except Exception`
    blocks in experiment code cannot swallow it (and write bad rows)."""


class GroqKeyPool:
    def __init__(self, keys=None, max_attempts=60, default_cooldown=65.0, max_wait=120.0):
        keys = keys or [k.strip().strip("\"'") for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
        if not keys:
            raise ValueError("No Groq API keys configured. Set GROQ_API_KEYS=key1,key2,...")
        self.keys = keys
        self.clients = [Groq(api_key=k) for k in keys]
        print(f"[groq] {len(keys)} key(s) loaded: " + ", ".join(f"#{n + 1}=...{k[-4:]}" for n, k in enumerate(keys)), flush=True)
        self.cooldown_until = [0.0] * len(keys)   # TPM/RPM cooldowns
        self.daily_dead = [False] * len(keys)     # TPD/RPD exhausted this run
        self.invalid = [False] * len(keys)        # rejected with 401 (bad key): skipped
        self.idx = 0                              # sticky current key
        self.max_attempts = max_attempts
        self.default_cooldown = default_cooldown
        self.max_wait = max_wait
        self.lock = threading.Lock()

    @staticmethod
    def _is_daily(err):
        msg = str(err).lower()
        return "per day" in msg or "(tpd)" in msg or "(rpd)" in msg

    @staticmethod
    def _retry_after(err, default):
        try:
            return float(err.response.headers.get("retry-after", default))
        except Exception:
            return default

    def _pick(self):
        """Current key first, then in sequence (wrapping). None if all cooling."""
        n, now = len(self.keys), time.time()
        usable = [not (self.daily_dead[i] or self.invalid[i]) for i in range(n)]
        if not any(usable):
            if all(self.daily_dead[i] or not self.invalid[i] for i in range(n)):
                raise DailyLimitExhausted()
            bad = [i + 1 for i in range(n) if self.invalid[i]]
            raise RuntimeError(f"No usable Groq keys: key(s) {bad} rejected as invalid (401), "
                               "the rest are daily-exhausted. Check GROQ_API_KEYS.")
        for off in range(n):
            i = (self.idx + off) % n
            if usable[i] and self.cooldown_until[i] <= now:
                self.idx = i
                return i
        return None

    def call(self, fn):
        """fn(client) -> result. Retries the same request until success,
        all keys daily-exhausted, or max_attempts."""
        attempts = 0
        while attempts < self.max_attempts:
            with self.lock:
                i = self._pick()
                sleep_for = 0.0
                if i is None:  # all live keys cooling: wait, don't spin
                    live = [self.cooldown_until[j] for j in range(len(self.keys)) if not (self.daily_dead[j] or self.invalid[j])]
                    sleep_for = min(max(min(live) - time.time(), 1.0), self.max_wait) + random.uniform(0, 2)
                    print(f"[groq] all keys rate-limited; waiting {sleep_for:.0f}s", flush=True)
            if sleep_for:
                time.sleep(sleep_for)
                continue

            attempts += 1
            try:
                return fn(self.clients[i])
            except RateLimitError as e:
                with self.lock:
                    if self._is_daily(e):
                        self.daily_dead[i] = True
                        print(f"[groq] key #{i + 1} hit DAILY limit", flush=True)
                    else:
                        self.cooldown_until[i] = time.time() + self._retry_after(e, self.default_cooldown)
                        print(f"[groq] key #{i + 1} rate-limited (TPM/RPM); rotating", flush=True)
                    self.idx = (i + 1) % len(self.keys)
            except AuthenticationError:
                with self.lock:
                    self.invalid[i] = True
                    print(f"[groq] key #{i + 1} (...{self.keys[i][-4:]}) is INVALID (401); skipping it", flush=True)
                    self.idx = (i + 1) % len(self.keys)
            except (APIConnectionError, APIStatusError) as e:
                if isinstance(e, APIStatusError) and e.status_code < 500:
                    raise
                time.sleep(min(2 ** min(attempts, 5), 30))  # transient: backoff, same key
        raise RuntimeError(f"Groq request failed after {self.max_attempts} attempts")


_pool = None


def get_pool(keys=None):
    """Process-wide singleton, so cooldown/daily-limit state is shared by every caller.
    `keys` only matters on the first call."""
    global _pool
    if _pool is None:
        _pool = GroqKeyPool(keys=keys)
    return _pool