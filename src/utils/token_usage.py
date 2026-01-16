TOKEN_PRICE_UNIT = 1000000

TOKEN_PRICE_MAP = {
    'gpt-4o' : {
        'input' : 2.5 / TOKEN_PRICE_UNIT,
        'output' : 10 / TOKEN_PRICE_UNIT
    },
    'gpt-4o-mini' : {
        'input' : 0.15 / TOKEN_PRICE_UNIT,
        'output' : 0.6 / TOKEN_PRICE_UNIT
    }
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