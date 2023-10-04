from fastchat.model import get_conversation_template


answer_prefix = "Answer:"

def doc_to_text_custom(doc):
    conv = get_conversation_template("lmsys/vicuna-7b-v1.3")

    message = (
        "Consider the sentence below in triple backticks "
        "and corresponding question. Does the sentence contain enough information "
        "to answer the question? Your answer should be either yes or no.\n\n"
        "Desired format:\n"
        "Answer: <your_answer>\n"
        f"Do not print \"{answer_prefix}\" again, just what you think the answer is.\n\n"
        f"Sentence:\n```\n{doc['sentence']}\n```\n"
        f"Question: {doc['question']}?\n"
        f"{answer_prefix}"
    )

    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()


def doc_to_text(doc):
    conv = get_conversation_template("lmsys/vicuna-7b-v1.3")

    message = (
        f"{doc['question']}\n{doc['sentence']}\n"
        "Question: Does this response answer the question?\n"
        "Answer:"
    )

    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)

    return conv.get_prompt()
