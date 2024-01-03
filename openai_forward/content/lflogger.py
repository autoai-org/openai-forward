import time
from typing import List
import orjson
from orjson import JSONDecodeError
from loguru import logger
from fastapi import Request
from ..helper import get_client_ip, get_unique_id, route_prefix_to_str
from ..settings import DEFAULT_REQUEST_CACHING_VALUE
from .helper import markdown_print, parse_sse_buffer, print
from langfuse import Langfuse

class LangfuseLogger:
    def __init__(self) -> None:
        self.langfuse = Langfuse()
        self.traces = {}
    
    
    @staticmethod
    async def parse_payload(request: Request):
        """
        Asynchronously parse the payload from a FastAPI request.

        Args:
            request (Request): A FastAPI request object.

        Returns:
            dict: A dictionary containing parsed messages, model, IP address, UID, and datetime.
        """
        uid = get_unique_id()
        payload = await request.json()

        # functions = payload.get("functions") # deprecated
        # if functions:
        #     info = {
        #         "functions": functions, # Deprecated in favor of `tools`
        #         "function_call": payload.get("function_call", None), # Deprecated in favor of `tool_choice`
        #     }
        info = {}
        info.update(
            {
                "messages": payload["messages"],
                "model": payload["model"],
                "stream": payload.get("stream", False),
                "max_tokens": payload.get("max_tokens", None),
                "response_format": payload.get("response_format", None),
                "n": payload.get("n", 1),
                "temperature": payload.get("temperature", 1),
                "top_p": payload.get("top_p", 1),
                "logit_bias": payload.get("logit_bias", None),
                "frequency_penalty": payload.get("frequency_penalty", 0),
                "presence_penalty": payload.get("presence_penalty", 0),
                "seed": payload.get("seed", None),
                "stop": payload.get("stop", None),
                "user": payload.get("user", None),
                "tools": payload.get("tools", None),
                "tool_choice": payload.get("tool_choice", None),
                "ip": get_client_ip(request) or "",
                "uid": uid,
                "caching": payload.pop(
                    "caching", DEFAULT_REQUEST_CACHING_VALUE
                ),  # pop caching
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
        )
        return info, orjson.dumps(payload)

    def parse_bytearray(self, buffer: bytearray):
        """
        Parses a bytearray, usually from an API response, into a dictionary containing various information.

        Args:
            buffer (List[bytes]): A list of bytes to parse.

        Returns:
            Dict[str, Any]: A dictionary containing metadata and content. The keys include:
                - "assistant" (str): content
                - "is_tool_calls" (boolean)
        """
        start_token = "data: "
        start_token_len = len(start_token)
        if buffer.startswith(b'data: '):
            txt_lines = parse_sse_buffer(buffer)
            stream = True
            first_dict = orjson.loads(txt_lines[0][start_token_len:])
            # todo: multiple choices
            msg = first_dict["choices"][0]["delta"]
        else:
            stream = False
            first_dict = orjson.loads(buffer)
            # todo: multiple choices
            msg = first_dict["choices"][0]["message"]

        target_info = dict()
        target_info["created"] = first_dict["created"]
        target_info["id"] = first_dict["id"]
        target_info["model"] = first_dict["model"]
        target_info["role"] = msg["role"]
        role = msg["role"]  # always be "assistant"
        content, tool_calls = msg.get("content"), msg.get("tool_calls")
        if tool_calls:
            """
            tool_calls:
                [{
                "index": 0,
                "id": 'xx',
                'type": 'function',
                'function': {'name': 'xxx', 'arguments': ''}
                 }]
            """
            target_info[role] = tool_calls
            target_info["is_tool_calls"] = True
            parse_content_key = "tool_calls"
        else:
            target_info[role] = content
            target_info["is_tool_calls"] = False
            parse_content_key = "content"

        if not stream:
            return target_info

        # loop for stream
        stream_content = ""
        for line in txt_lines[1:]:
            if line.startswith(start_token):
                delta_content, usage = self._parse_one_line_content(
                    line[start_token_len:], parse_content_key
                )
                stream_content += delta_content
                if usage:
                    target_info["usage"] = usage
                    print(target_info)
        if target_info['is_tool_calls']:
            tool_calls[0]['function']['arguments'] = stream_content
            target_info[role] = tool_calls
        else:
            target_info[role] = stream_content
        return target_info

    @staticmethod
    def _parse_one_line_content(line: str, parse_key="content"):
        """
        Helper method to parse a single line.

        Args:
            line (str): The line to parse.
            parse_key (str): .

        Returns:
            str: The parsed content from the line.
        """
        try:
            line_dict = orjson.loads(line)
            if parse_key == "content":
                usage = None
                delta_content = line_dict["choices"][0]["delta"][parse_key]
                finish_reason = line_dict["choices"][0]['finish_reason']
                if finish_reason == "stop":
                    usage = line_dict['usage']
                return delta_content, usage
            elif parse_key == "tool_calls":
                tool_calls = line_dict["choices"][0]["delta"]["tool_calls"]
                return tool_calls[0]["function"]['arguments'], None
            else:
                logger.error(f"Unknown parse key: {parse_key}")
                return "", None
        except JSONDecodeError:
            return "", None
        except KeyError:
            return "", None
    
    async def start(self, uid, request: Request):
        info, _ = await self.parse_payload(request)
        trace = self.langfuse.trace(
            id=uid,
        )
        metadata = info.copy()
        metadata.pop('messages')
        generation = trace.generation(
            id=uid,
            name='chat.completion',
            input=info['messages'],
            model = info['model'],
            metadata=metadata
        )
        self.traces[uid] = (trace, generation)
        
    def end(self, uid, result):
        trace, generation = self.traces[uid]
        logger.info(result)
        generation.end(output=result['assistant'])
        trace.update(output=result['assistant'])
        self.langfuse.flush()

lfLogger = LangfuseLogger()