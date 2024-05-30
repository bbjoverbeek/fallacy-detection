from prompt_model import extract_model_answer

fallacy_options = [
    'slippery slope', 'ad hominem', 'appeal to (false) authority',
    'X appeal to majority', 'no fallacy'
]

test_texts = [
    """### Response:
    I believe this argument contains a , difference but I am not entirely sure which one. appeal to authority It seems to mislead the audience in some way."""
]

for text in test_texts:
    answer = extract_model_answer(text, fallacy_options)
    print(f"Extracted Answer: {answer}")
