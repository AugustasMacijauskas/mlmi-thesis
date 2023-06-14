def replace_text_with_whitespace(text, new_text):
    # Extract leading and trailing whitespace
    leading_whitespace = text[:len(text) - len(text.lstrip())]
    trailing_whitespace = text[len(text.rstrip()):]

    # Create the modified string with preserved whitespace
    modified_text = leading_whitespace + new_text + trailing_whitespace

    return modified_text


def combine_strings_with_whitespace(left_string, right_string):
    # Calculate the length of whitespace to extract
    whitespace_length = len(left_string) - len(left_string.lstrip())

    # Extract the whitespace
    whitespace = left_string[:whitespace_length]

    # Combine the strings with the whitespace
    combined_string = right_string + whitespace

    return combined_string


def main():
    # Example usage of replace_text_with_whitespace
    original_text = "   Hello World!   "
    new_text = "Hi, OpenAI!"

    modified_text = replace_text_with_whitespace(original_text, new_text)
    print(f"\"{modified_text}\"")  # "   Hi, OpenAI!   "


    # Example usage of combine_strings_with_whitespace
    left_string = "    Hello"
    right_string = "World!"

    combined_string = combine_strings_with_whitespace(left_string, right_string)
    print(f"\"{combined_string}\"") # "World!    "


if __name__ == "__main__":
    main()