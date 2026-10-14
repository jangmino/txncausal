"""Vendored VERBATIM from msparkDev/ConcurrentBERT-Recommender
scripts/BERT/DataPrepWith.py (text-pair construction only) -- Park & Oh (2024)
baseline. Do not edit: fidelity of the baseline's text protocol depends on these
being byte-identical (sole changes: dropped their module-level imports/main flow;
removed one debug print in compile_order_history's truncation path).
Functions: format_next_purchase / get_longer_text / add_text_if_fits /
compile_order_history. All take the tokenizer as an argument.
"""
import pandas as pd

# Function to format the next purchase prediction for output.
def format_next_purchase(user_text):
    """
    Formats the text for the next purchase prediction.

    Parameters:
    - user_text: DataFrame or Series containing the next purchase's description.

    Returns:
    - item_text: Formatted string for the next purchase prediction.
    """
    try:
        if pd.notna(user_text.attribute_values.iloc[0]):
            item_description = f'{user_text.category_name.iloc[0]} - {user_text.product_name.iloc[0]} {user_text.attribute_values.iloc[0]}'
        else:
            item_description = f'{user_text.category_name.iloc[0]} - {user_text.product_name.iloc[0]}'

    except IndexError:
        if pd.notna(user_text.attribute_values):
            item_description = f'{user_text.category_name} - {user_text.product_name} {user_text.attribute_values}'
        else:
            item_description = f'{user_text.category_name} - {user_text.product_name}'
    except AttributeError:
        if pd.notna(user_text.attribute_values):
            item_description = f'{user_text.category_name} - {user_text.product_name} {user_text.attribute_values}'
        else:
            item_description = f'{user_text.category_name} - {user_text.product_name}'

    item_text = f'## 다음 구매 상품 예측\n{item_description}'
    return item_text


# Function to determine the longer text between two given texts based on token count.
def get_longer_text(text_1, text_2, tokenizer):
    """
    Determines the longer text between two texts based on token count.

    Parameters:
    - text_1: First text to compare.
    - text_2: Second text to compare.
    - tokenizer: Tokenizer object used for tokenizing the texts.

    Returns:
    - text_max: The longer text of the two inputs.
    """
    text_max = text_1 if len(tokenizer(text_1, padding=True, truncation=False).input_ids) > \
                         len(tokenizer(text_2, padding=True, truncation=False).input_ids) else text_2
    return text_max


# Function to attempt adding additional text without exceeding token limit.
def add_text_if_fits(current_text, addition, tokenizer, item_max):
    """
    Attempts to add additional text to the given text without exceeding the token limit.

    Parameters:
    - current_text: The current text string to which addition is attempted.
    - addition: The text to add to the current text.
    - tokenizer: Tokenizer object used for tokenizing the texts.
    - item_max: Maximum item size for consideration in tokenization.

    Returns:
    - new_text or None: The new text string with the addition or None if exceeding token limit.
    """
    new_text = current_text + addition
    if len(tokenizer(new_text, item_max, padding=True, truncation=False).input_ids) < 512:
        return new_text
    else:
        return None


# Function to compile user's order history.
def compile_order_history(user_prompt, tokenizer, item_max):
    """
    Compiles the order history for a user into a text string.

    Parameters:
    - user_prompt: DataFrame containing the user's orders except the last.
    - tokenizer: Tokenizer object used for tokenizing the texts.
    - item_max: Maximum item size for consideration in tokenization.

    Returns:
    - user_text: Compiled order history as a text string.
    """
    user_text = "## 주문 히스토리"
    user_prompt_grouped = user_prompt.groupby('initial_paid_at', sort=False)

    ctemp = ''
    for date, group in user_prompt_grouped:
        temp = f'\n{date}의 주문 내역'
        if len(group) > 1:
            ctemp += temp + ' [동시구매]'
        if pd.notna(group.iloc[0].attribute_values):
            ctemp = ctemp + f': {group["category_name"].iloc[0]} - {group["product_name"].iloc[0]} {group["attribute_values"].iloc[0]}'
        else:
            ctemp = ctemp + f': {group["category_name"].iloc[0]} - {group["product_name"].iloc[0]}'

        new_user_text = add_text_if_fits(user_text, ctemp, tokenizer, item_max)
        if new_user_text is not None:
            temp = (ctemp + ': ')
        else:
            user_text += temp
            return user_text

        len_group = len(group)
        count = 0
        for index, row in group.iterrows():
            if count > 0:
                ctemp += ', '
            else:
                ctemp += ''
            count += 1
            if pd.notna(row.attribute_values):
                ctemp += f'{row["category_name"]} - {row["product_name"]} {row["attribute_values"]}'
            else:
                ctemp += f'{row["category_name"]} - {row["product_name"]}'

            new_user_text = add_text_if_fits(user_text, ctemp, tokenizer, item_max)
            if new_user_text is not None:
                temp = ctemp
                continue
            else:
                user_text += temp
                return user_text

    return user_text


