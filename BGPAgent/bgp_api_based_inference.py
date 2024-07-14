import os
import json
from tqdm import tqdm
from groq import Groq
from openai import OpenAI
import re
from prompt import *

openai_client = OpenAI(
    api_key=os.environ.get("AI_HUB_MIX_API_KEY"),
    base_url="https://aihubmix.com/v1"
)
groq_client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://groq.huggingtiger.asia/",
)


system_prompt = f"""{zero_shot_system_prompt}"""

user_content_list = []
with open("./program_data/type1_input.json", "r") as f:
    user_content_list = json.load(f)

output_list = []

pattern = re.compile(r'\[.*?\]', re.DOTALL)


def get_client_chat_completion(client, model_name, messages, args):
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        **args
    )
    return chat_completion.choices[0].message.content

def update_user_content(user_content, model_name, value):
        user_content[model_name+"-answer"] = value
        user_content[model_name+"-answer-list"] = pattern.findall(value)
        return user_content


for user_content in tqdm(user_content_list):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_content['question'],
        }]
    args = {
        "temperature": 0.0,  # Optional
    }
    llama3_70b_model_name = "llama3-70b-8192"
    llama3_70b_value = get_client_chat_completion(groq_client, llama3_70b_model_name, messages, args)
    user_content = update_user_content(user_content, llama3_70b_model_name, llama3_70b_value)

    # llama3_8b_chat_completion = groq_client.chat.completions.create(
    #     model= "llama3-8b-8192",
    #     messages=message,
    #     temperature=0.0  # Optional
    #     )
    # llama3_8b_output=llama3_8b_chat_completion.choices[0].message.content
    # user_content["llama3-8b-answer"] = llama3_8b_output
    llam3_8b_value = get_client_chat_completion(groq_client, "llama3-8b-8192", messages, args)
    user_content = update_user_content(user_content, "llama3-8b-8192", llam3_8b_value)

    mistral_8_7b_value = get_client_chat_completion(groq_client, "mixtral-8x7b-32768", messages, args)
    user_content = update_user_content(user_content, "mixtral-8x7b-32768", mistral_8_7b_value)

    gemma_7b_value = get_client_chat_completion(groq_client, "gemma-7b-it", messages, args)
    user_content = update_user_content(user_content, "gemma-7b-it", gemma_7b_value)

    gpt_3__5_turbo_value = get_client_chat_completion(openai_client, "gpt-3.5-turbo", messages, args)
    user_content = update_user_content(user_content, "gpt-3.5-turbo", gpt_3__5_turbo_value)

    gpt_4_turbo_value = get_client_chat_completion(openai_client, "gpt-4-turbo", messages, args)
    user_content = update_user_content(user_content, "gpt-4-turbo", gpt_4_turbo_value)

    gpt_4o_value = get_client_chat_completion(openai_client, "gpt-4o", messages, args)
    user_content = update_user_content(user_content, "gpt-4o", gpt_4o_value)

    claude_3__5_sonnet_value = get_client_chat_completion(openai_client, "claude-3-5-sonnet-20240620", messages, args)
    user_content = update_user_content(user_content, "claude-3-5-sonnet-20240620", claude_3__5_sonnet_value)

    # print(user_content)
    output_list.append(user_content)
    with open("./program_data/cache/cache.json", "w", encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

with open("./program_data/type1_english_output_0714_add_more_model_zero_shot_temperation=0.json", "w", encoding='utf-8') as f:
    json.dump(output_list, f, ensure_ascii=False, indent=4)
