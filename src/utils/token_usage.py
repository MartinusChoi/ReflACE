from typing import Dict

from langchain.messages import AIMessage

TOKEN_PRICE_UNIT = 1000000

TOKEN_PRICE_MAP = {
    'gpt-4o' : {
        'input' : 2.5 / TOKEN_PRICE_UNIT,
        'output' : 10 / TOKEN_PRICE_UNIT
    },
    'gpt-4o-mini' : {
        'input' : 0.15 / TOKEN_PRICE_UNIT,
        'output' : 0.6 / TOKEN_PRICE_UNIT
    },
    'gpt-4.1-mini' : {
        'input' : 0.40 / TOKEN_PRICE_UNIT,
        'output' : 1.60 / TOKEN_PRICE_UNIT
    },
}


def calc_token_price(
    model:str, 
    input_tokens:int, 
    output_tokens:int
):
    input_token_price = input_tokens * TOKEN_PRICE_MAP[model]['input']
    output_token_price = output_tokens * TOKEN_PRICE_MAP[model]['output']
    total_token_price = input_token_price + output_token_price

    return {
        'input_token_price' : input_token_price,
        'output_token_price' : output_token_price,
        'total_token_price' : total_token_price
    }

def get_token_usage_from_message(message: AIMessage) -> Dict[str, int]:

    try:
        input_tokens = message.usage_metadata['input_tokens']
        output_tokens = message.usage_metadata['output_tokens']
        total_tokens = message.usage_metadata['total_tokens']
    except Exception as error:
        raise error
    
    return {
        'input_tokens' : input_tokens,
        'output_tokens' : output_tokens,
        'total_tokens' : total_tokens
    }