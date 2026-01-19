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
            print(f"\n[Actor] ✅ Request succeed on attept {attempt+1}/{max_retries}")
            break
        except Exception as error:
            print(f"[Actor] ⚠️ Request failed on attept {attempt+1}/{max_retries}")
            # raise error when attempt hit max retry limit.
            if attempt + 1 == max_retries:
                print(f"[Actor] ⛔️ Model Request failed. Please Try Later.")
                raise error
            
    return response
