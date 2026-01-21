from typing import Sequence

from langchain.messages import AIMessage, AnyMessage

from langchain_openai import ChatOpenAI

def get_response_with_retry(
    model_client: ChatOpenAI, 
    messages: Sequence[AnyMessage], 
    max_retries: int
) -> AIMessage:

    for attempt in range(max_retries):
        try:
            response: AIMessage = model_client.invoke(messages)
            break
        except Exception as error:
            # raise error when attempt hit max retry limit.
            if attempt + 1 == max_retries:
                raise error
            
    return response
