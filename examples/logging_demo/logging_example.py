"""Logging demo -- cat-agent's loguru logger with an agent.

Logging is configured entirely via environment variables -- no CLI flags needed.

    # coloured INFO output (default)
    CAT_AGENT_LOG_LEVEL=INFO  python examples/logging_demo/logging_example.py

    # full DEBUG verbosity
    CAT_AGENT_LOG_LEVEL=DEBUG python examples/logging_demo/logging_example.py

    # structured JSON logs
    CAT_AGENT_LOG_LEVEL=INFO CAT_AGENT_LOG_FORMAT=json python examples/logging_demo/logging_example.py

    # also write to a rotating file
    CAT_AGENT_LOG_LEVEL=DEBUG CAT_AGENT_LOG_FILE=agent.log python examples/logging_demo/logging_example.py
"""

from __future__ import annotations

import os
import time

from cat_agent.agents import Assistant
from cat_agent.log import logger, setup_logger
from cat_agent.tools.base import BaseTool, register_tool

@register_tool("get_weather")
class GetWeather(BaseTool):
    description = (
        "Get the current weather for a city. "
        "Provide the 'city' name as a string."
    )
    parameters = {
        "type": "object",
        "properties": {
            "city": {
                "description": "Name of the city to get the weather for",
                "type": "string",
            },
        },
        "required": ["city"],
    }

    def call(self, params: str, **kwargs) -> str:
        params = self._verify_json_format_args(params)
        city = params.get("city", "unknown")

        logger.info("Weather lookup requested for city: {}", city)

        # Simulate a slow API call
        time.sleep(0.3)
        logger.debug("Weather API responded for {}", city)

        return f"The weather in {city} is 22 C and sunny."


def main() -> None:
    # Enable logging by default for this demo.  If CAT_AGENT_LOG_LEVEL is
    # already set the logger was configured at import time, so we skip this.
    if not os.environ.get("CAT_AGENT_LOG_LEVEL"):
        setup_logger(level="DEBUG")

    llm_cfg = {
        "model_type": "llama_cpp",
        "repo_id": "Salesforce/xLAM-2-3b-fc-r-gguf",
        "filename": "xLAM-2-3B-fc-r-F16.gguf",
        "n_ctx": 4096,
        "n_gpu_layers": -1,
        "n_threads": 6,
        "temperature": 0.6,
        "max_tokens": 1024,
        "verbose": False,
    }
    logger.debug("LLM config: {}", llm_cfg)

    bot = Assistant(
        llm=llm_cfg,
        name="WeatherBot",
        description="An agent that can look up the weather for any city.",
        function_list=["get_weather"],
    )
    logger.info("Agent '{}' created with tools: {}", bot.name, list(bot.function_map.keys()))

    user_query = "What is the weather in Istanbul?"
    messages = [{"role": "user", "content": user_query}]

    logger.info("Sending user query: {}", user_query)
    print(f"\nUser: {user_query}")
    print("-" * 40)

    response = []
    for response in bot.run(messages=messages):
        logger.debug("Received streamed chunk ({} messages so far)", len(response))

    if response:
        final = response[-1].get("content", "")
        logger.info("Agent response: {}", final)
        print(f"\nWeatherBot: {final}")
    else:
        logger.warning("Agent returned an empty response")
        print("\nWeatherBot: (no response)")

    logger.info("Done!")


if __name__ == "__main__":
    main()
