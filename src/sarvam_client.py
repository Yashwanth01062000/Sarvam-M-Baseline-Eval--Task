import requests


class SarvamClient:
    """
    Client for calling the Sarvam Chat Completions API.
    """

    def __init__(
        self,
        api_key,
        api_url,
        model,
        timeout=120,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    @staticmethod
    def _extract_content(data):
        """
        Extract assistant message content
        from the Sarvam API response.
        """

        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""

    def chat(
        self,
        prompt,
        temperature=0,
    ):
        """
        Send a prompt to the Sarvam Chat Completions API
        and return the generated output and token usage.
        """

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "temperature": temperature,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        output = self._extract_content(data)

        usage = data.get(
            "usage",
            {}
        )

        return {
            "output": output,
            "usage": usage,
            "raw": data,
        }
