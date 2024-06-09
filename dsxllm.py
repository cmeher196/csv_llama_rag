import time
import requests

from typing import Any, List, Mapping, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun

requests.packages.urllib3.disable_warnings()


class DsxLLM(LLM):
    model_name: str
    api_keys: List[str]
    cur_key_idx: Optional[int] = 0
    delay_sec: int = 5
    last_called: Optional[int] = -1

    @property
    def _llm_type(self) -> str:
        return f"custom-{self.model_name}"

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_name": self.model_name,
                "api_keys": "..."}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        if stop is not None:
            pass
            # raise ValueError("stop kwargs are not permitted.")
        while self.get_wait_seconds() > 0:
            wait_seconds = self.get_wait_seconds()
            print(
                f"Waiting additional {wait_seconds:.1f} sec between LLM calls...")
            time.sleep(wait_seconds)
        self.last_called = time.time()
        # print("Calling LLM ...")
        return self.llm_api(prompt)

    def get_wait_seconds(self):
        if self.last_called > 0:
            wait_seconds = max(
                self.delay_sec + self.last_called - time.time(), 0)
        else:
            wait_seconds = 0
        return wait_seconds

    def llm_api(self, instruction):
        """
        Creates a request to LLM model with API key in header.
        """

        url = f"https://opensource-llm-api.aiaccel.dell.com/llm/{self.model_name}"
        instr_len = len(instruction)

        payload = {"instruction": instruction}

        current_key = self.api_keys[self.cur_key_idx]

        headers = {
            'accept': 'application/json',
            'api-key': current_key,
            'Content-Type': 'application/json'
        }

        try:
            response = requests.post(
                url, headers=headers, json=payload, verify=False)
            self.cur_key_idx = (self.cur_key_idx + 1) % len(self.api_keys)
            response.raise_for_status()  # Raise exception for 4xx and 5xx status codes
            response_dict = response.json()
            generated_text = response_dict['response'][0]['generated_text']
            if self.model_name == 'falcon-40b-instruct':
                # to remove the instruction from the generated text
                generated_text = generated_text[instr_len:]
            return generated_text
        except Exception as err:
            if response is not None:
                print("Error code:", response.status_code)
                print("Error message:", response.text)
            else:
                print("Error:", err)
            return None
