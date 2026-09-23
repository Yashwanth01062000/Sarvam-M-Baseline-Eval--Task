import time
import requests

from .config import (
    SARVAM_API_KEY,
    SARVAM_API_URL,
    SARVAM_MODEL,
    SARVAM_TEMPERATURE,
    SARVAM_TIMEOUT,
)


class SarvamAPIError(RuntimeError):
    pass


class SarvamClient:
    def __init__(self):
        if not SARVAM_API_KEY:
            raise SarvamAPIError(
                "SARVAM_API_KEY is not configured. "
                "Put it in .env or set it in PowerShell."
            )

        self.session = requests.Session()

        self.headers = {
            "api-subscription-key": SARVAM_API_KEY,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_content(data):
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise SarvamAPIError(
                f"Unexpected API response structure: {data}"
            ) from exc

    def chat(self, prompt, retries=3):
        payload = {
            "model": SARVAM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": SARVAM_TEMPERATURE,
        }

        last_error = None

        for attempt in range(retries + 1):
            start = time.perf_counter()

            try:
                response = self.session.post(
                    SARVAM_API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=SARVAM_TIMEOUT,
                )

                latency = time.perf_counter() - start

                # Retry rate-limit and server errors
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(
                        f"Retryable HTTP status {response.status_code}: "
                        f"{response.text[:1000]}"
                    )

                # Show the actual API error for other HTTP errors
                if not response.ok:
                    raise requests.HTTPError(
                        f"HTTP {response.status_code}: "
                        f"{response.text[:2000]}"
                    )

                data = response.json()

                output = self._extract_content(data)

                usage = data.get("usage") or {}

                return {
                    "output": output,
                    "latency_s": round(latency, 3),
                    "input_tokens": usage.get(
                        "prompt_tokens",
                        usage.get("input_tokens", ""),
                    ),
                    "output_tokens": usage.get(
                        "completion_tokens",
                        usage.get("output_tokens", ""),
                    ),
                    "total_tokens": usage.get(
                        "total_tokens",
                        "",
                    ),
                    "model": SARVAM_MODEL,
                    "status": "success",
                    "error": "",
                }

            except (
                requests.RequestException,
                ValueError,
                SarvamAPIError,
            ) as exc:

                last_error = exc

                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    latency = time.perf_counter() - start

        return {
            "output": "",
            "latency_s": round(latency, 3),
            "input_tokens": "",
            "output_tokens": "",
            "total_tokens": "",
            "model": SARVAM_MODEL,
            "status": "error",
            "error": str(last_error),
        }