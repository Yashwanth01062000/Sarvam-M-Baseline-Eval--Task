import requests


class SarvamClient:

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

    def chat(
        self,
        prompt,
        temperature=0,
    ):
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

        output = (
            data["choices"][0]["message"]
            .get("content", "")
        )

        usage = data.get("usage", {})

        return {
            "output": output,
            "usage": usage,
            "raw": data,
        }
