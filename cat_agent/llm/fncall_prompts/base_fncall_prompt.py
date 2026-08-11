# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Literal, Union

from cat_agent.llm.schema import Message


class BaseFnCallPrompt(object):

    @staticmethod
    def preprocess_fncall_messages(messages: List[Message],
                                   functions: List[dict],
                                   lang: Literal['en', 'zh'],
                                   function_choice: Union[Literal['auto'], str] = 'auto',
                                   **kwargs) -> List[Message]:
        """
        Preprocesss the messages and add the function calling prompt,
        assuming the input and output messages are in the multimodal format.
        """
        assert function_choice != 'none'
        raise NotImplementedError

    @staticmethod
    def postprocess_fncall_messages(messages: List[Message],
                                    function_choice: Union[Literal['auto'], str] = 'auto',
                                    **kwargs) -> List[Message]:
        """
        Transform the plaintext model output into structured function call messages,
        return in the multimodal format for consistency.
        """
        raise NotImplementedError
