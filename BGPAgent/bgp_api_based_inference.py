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
    
    llama3_70b_chat_completion = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.0  # Optional
    )

    # llama3_8b_chat_completion = groq_client.chat.completions.create(
    #     model= "llama3-8b-8192",
    #     messages=message,
    #     temperature=0.0  # Optional
    #     )

    # mistral_8_7b_chat_completion = groq_client.chat.completions.create(
    #     model= "mixtral-8x7b-32768",
    #     messages=message,
    #     temperature=0.0  # Optional
    #     )

    # gemma_7b_chat_completion = groq_client.chat.completions.create(
    #     model= "gemma-7b-it",
    #     messages=message,
    #     temperature=0.0  # Optional
    #     )

    gpt_3__5_chat_completion = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.0,
    )

    gpt_4o_chat_completion = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.0,
    )

    gpt_4_turbo_chat_completion = openai_client.chat.completions.create(
        model="gpt-4-turbo",
        messages=messages,
        temperature=0.0,
    )

    claude_3__5_sonnet_chat_completion = openai_client.chat.completions.create(
        model="claude-3-5-sonnet-20240620",
        messages=messages,
        temperature=0.0,
    )
    pattern = re.compile(r'\[.*?\]', re.DOTALL)

    llama3_70b_output = llama3_70b_chat_completion.choices[0].message.content
    user_content["llama3-70b-answer"] = llama3_70b_output
    user_content["llama3-70b-answer-list"] = pattern.findall(llama3_70b_output)
    
    # llama3_8b_output=llama3_8b_chat_completion.choices[0].message.content
    # user_content["llama3-8b-answer"] = llama3_8b_output
    
    # mistral_8_7b_output=mistral_8_7b_chat_completion.choices[0].message.content
    # user_content["mistral-8-7b-answer"] = mistral_8_7b_output
    
    # gemma_7b_output=gemma_7b_chat_completion.choices[0].message.content
    # user_content["gemma-7b-answer"] = gemma_7b_output

    gpt_3__5_turbo_output = gpt_3__5_chat_completion.choices[0].message.content
    user_content["gpt-3.5-turbo-answer"] = gpt_3__5_turbo_output
    user_content["gpt-3.5-turbo-answer-list"] = pattern.findall(gpt_3__5_turbo_output)

    gpt_4o_output = gpt_4o_chat_completion.choices[0].message.content
    user_content["gpt-4o-answer"] = gpt_4o_output
    user_content["gpt-4o-answer-list"] = pattern.findall(gpt_4o_output)

    gpt_4_turbo_output = gpt_4_turbo_chat_completion.choices[0].message.content
    user_content["gpt-4-turbo-answer"] = gpt_4_turbo_output
    user_content["gpt-4-turbo-answer-list"] = pattern.findall(gpt_4_turbo_output)

    claude_3__5_output = claude_3__5_sonnet_chat_completion.choices[0].message.content
    user_content["claude-3-5-sonnet-20240620-answer"] = claude_3__5_output
    user_content["claude-3-5-sonnet-20240620-answer-list"] = pattern.findall(claude_3__5_output)

    # print(user_content)
    output_list.append(user_content)
    with open("./program_data/cache/cache.json", "w", encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)

with open("./program_data/type1_english_output_0713_zero_shot_temperation=0.json", "w", encoding='utf-8') as f:
    json.dump(output_list, f, ensure_ascii=False, indent=4)
