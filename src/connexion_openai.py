# import os
# import openai
#
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']
# print(openai.api_key)


import os
def check_api_exists():
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print(f"OPENAI_API_KEY is set")
    else:
        print("OPENAI_API_KEY is not set.")

if __name__ == '__main__':
    check_api_exists()