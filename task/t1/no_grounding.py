import asyncio
from asyncio import gather
from typing import Any
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from openai import azure_endpoint
from pydantic import SecretStr
from task._constants import DIAL_URL, API_KEY
from task.user_client import UserClient

#TODO:
# Before implementation open the `flow_diagram.png` to see the flow of app

BATCH_SYSTEM_PROMPT = """You are a user search assistant. Your task is to find users from the provided list that match the search criteria.

INSTRUCTIONS:
1. Analyze the user question to understand what attributes/characteristics are being searched for
2. Examine each user in the context and determine if they match the search criteria
3. For matching users, extract and return their complete information
4. Be inclusive - if a user partially matches or could potentially match, include them

OUTPUT FORMAT:
- If you find matching users: Return their full details exactly as provided, maintaining the original format
- If no users match: Respond with exactly "NO_MATCHES_FOUND"
- If uncertain about a match: Include the user with a note about why they might match"""

FINAL_SYSTEM_PROMPT = """You are a helpful assistant that provides comprehensive answers based on user search results.

INSTRUCTIONS:
1. Review all the search results from different user batches
2. Combine and deduplicate any matching users found across batches
3. Present the information in a clear, organized manner
4. If multiple users match, group them logically
5. If no users match, explain what was searched for and suggest alternatives"""

USER_PROMPT = """## USER DATA:
{context}

## SEARCH QUERY: 
{query}"""


class TokenTracker:
    def __init__(self):
        self.total_tokens = 0
        self.batch_tokens = []

    def add_tokens(self, tokens: int):
        self.total_tokens += tokens
        self.batch_tokens.append(tokens)

    def get_summary(self):
        return {
            'total_tokens': self.total_tokens,
            'batch_count': len(self.batch_tokens),
            'batch_tokens': self.batch_tokens
        }

#TODO:
# 1. Create AzureChatOpenAI client
#    hint: api_version set as empty string if you gen an error that indicated that api_version cannot be None
# 2. Create TokenTracker

client = AzureChatOpenAI(
    temperature=0.0,
    azure_deployment="gpt-4o",
    azure_endpoint=DIAL_URL,
    api_key=SecretStr(API_KEY),
    api_version=""
)
token_tracker = TokenTracker()

def join_context(context: list[dict[str, Any]]) -> str:
    #TODO:
    # You cannot pass raw JSON with user data to LLM (" sign), collect it in just simple string or markdown.
    # You need to collect it in such way:
    # User:
    #   name: John
    #   surname: Doe
    #   ...

    user_data = ""
    for user in context:
        user_data += "User:\n"
        for k,v in user.items():
            user_data += f"\t{k}: {v}\n"
        user_data += "\n"
    return user_data


async def generate_response(system_prompt: str, user_message: str) -> str:
    print("Processing...")
    #TODO:
    # 1. Create messages array with system prompt and user message
    # 2. Generate response (use `ainvoke`, don't forget to `await` the response)
    # 3. Get usage (hint, usage can be found in response metadata (its dict) and has name 'token_usage', that is also
    #    dict and there you need to get 'total_tokens')
    # 4. Add tokens to `token_tracker`
    # 5. Print response content and `total_tokens`
    # 5. return response content

    system_message = SystemMessage(content=system_prompt)
    user_message = HumanMessage(content=user_message)

    message = [system_message, user_message]

    response = await client.ainvoke(input=message)
    response_content = response.content
    response_total_tokens = response.response_metadata["token_usage"]["total_tokens"]
    print(f"Content: {response_content}")
    print(f"Token usage: {response_total_tokens}")
    token_tracker.add_tokens(response_total_tokens)
    return response_content


async def main():
    print("Query samples:")
    print(" - Do we have someone with name John that loves traveling?")

    user_question = input("> ").strip()
    if user_question:
        print("\n--- Searching user database ---")

        #TODO:
        # 1. Get all users (use UserClient)
        users = UserClient().get_all_users()

        # 2. Split all users on batches (100 users in 1 batch). We need it since LLMs have its limited context window
        batches = [users[i:i + 100] for i in range(0, len(users), 100)]

        # 3. Prepare tasks for async run of response generation for users batches:
        #       - create array tasks
        #       - iterate through `user_batches` and call `generate_response` with these params:
        #           - BATCH_SYSTEM_PROMPT (system prompt)
        #           - User prompt, you need to format USER_PROMPT with context from user batch and user question
        tasks = []
        for batch in batches:
            system_prompt = BATCH_SYSTEM_PROMPT
            user_message = USER_PROMPT.format(context=join_context(batch), query=user_question)
            tasks.append(generate_response(system_prompt, user_message))

        # 4. Run task asynchronously, use method `gather` form `asyncio`
        results = await asyncio.gather(*tasks)

        # 5. Filter results on 'NO_MATCHES_FOUND' (see instructions for BATCH_SYSTEM_PROMPT)
        filtered_results = [result for result in results if result.strip() != "NO_MATCHES_FOUND"]

        # 5. If results after filtration are present:
        #       - combine filtered results with "\n\n" spliterator
        #       - generate response with such params:
        #           - FINAL_SYSTEM_PROMPT (system prompt)
        #           - User prompt: you need to make augmentation of retrieved result and user question
        if filtered_results:
            combined_results = "\n\n".join(filtered_results)

            final_response = await generate_response(
                system_prompt=FINAL_SYSTEM_PROMPT,
                user_message=f"SEARCH RESULTS:\n{combined_results}\n\nORIGINAL QUERY: {user_question}"
            )

            print(final_response)
        # 6. Otherwise prin the info that `No users found matching`
        else:
            print("No users found matching")

        # 7. In the end print info about usage, you will be impressed of how many tokens you have used. (imagine if we have 10k or 100k users 😅)
        summary = token_tracker.get_summary()
        print(f">>=== Token usage ===<<")
        print(f"Number of Batch LLM calls: {summary['batch_count']}")
        print(f"Total token usage: {summary['total_tokens']}")


if __name__ == "__main__":
    asyncio.run(main())


# The problems with No Grounding approach are:
#   - If we load whole users as context in one request to LLM we will hit context window
#   - Huge token usage == Higher price per request
#   - Added + one chain in flow where original user data can be changed by LLM (before final generation)
# User Question -> Get all users -> ‼️parallel search of possible candidates‼️ -> probably changed original context -> final generation