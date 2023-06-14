def replace_text_with_whitespace(text, new_text):
    # Extract leading and trailing whitespace
    leading_whitespace = text[:len(text) - len(text.lstrip())]
    trailing_whitespace = text[len(text.rstrip()):]

    # Create the modified string with preserved whitespace
    modified_text = leading_whitespace + new_text + trailing_whitespace

    return modified_text


def main():
    # Example usage
    original_text = "   Hello World!   "
    new_text = "Hi, OpenAI!"

    modified_text = replace_text_with_whitespace(original_text, new_text)
    print(modified_text)  # "   Hi, OpenAI!   "


if __name__ == "__main__":
    main()