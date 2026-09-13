import warnings
warnings.filterwarnings('ignore')

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

LLM_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Phi-4-mini-instruct",
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=False,
        )

LLM_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct", clean_up_tokenization_spaces=False)
pipe = pipeline("text-generation", model=LLM_model, tokenizer=LLM_tokenizer)
generation_args = {"max_new_tokens": 512, "return_full_text": False, "do_sample": False}

system_prompt = """
You must parse the given natural language input for the mission at hand. All inputs after the first will be updates on how the mission is going.
Your outputs must follow a specified JSON format.

{
    "movement_target": [int, int, int],
    "found_target": bool,
}

1.
"""

# TODO: Make new structure an array of the structure below. Each entry is its own bot.

messages = [{"role": "system", "content": system_prompt}]

def generate_response(message):
    messages.append({"role": "user", "content": message})
    response = pipe(messages, **generation_args)[0]['generated_text']
    messages.append({"role": "assistant", "content": f"{response.strip()}\n\n"})
    return response

if __name__ == "__main__":
    print("Type 'quit', 'q', or 'exit' to exit.")
    while True:
        user_input = input("\033[34mYou: ").strip()
        print("\033[0m")

        if user_input.lower() in ["quit", "q", "exit"]:
            break

        response = generate_response(user_input)
        print(f"\033[32mAI: {response}\033[0m")



# Saved functions from NeLV
"""
class Chatbot(QMainWindow):

    def generate_response(self, message, LLM_model, LLM_tokenizer):
        if self.mode in ["Short Range", "Medium Range", "Long Range"]:
            self.messages.append({"role": "user", "content": f"{message}\nOnly output the JSON object. No prefix, additional text, or explanation.\n\n"})
        else:
            self.messages.append({"role": "user", "content": message})
        pipe = pipeline("text-generation", model=LLM_model, tokenizer=LLM_tokenizer,)
        generation_args = {"max_new_tokens": 512, "return_full_text": False, "temperature": 0.0, "do_sample": False, }
        response = pipe(self.messages, **generation_args)[0]['generated_text']
        cleaned = response.strip().rstrip('\n')
        self.messages.append({"role": "assistant", "content": f"{cleaned}\n\n"})
        return response

    def parse_response(self, response):
        try:
            self.planned_flight = json.loads(response)
        except:
            try:
                if "{" in response and "}" in response:
                    last_open = response.rindex("{")
                    last_close = response.rindex("}")
                    response = response[last_open:last_close+1]
                try:
                    self.planned_flight = json.loads(response)
                except:
                    print("=" * 50)
                    print("Wrong JSON format!")
                    print(response)
                    print("=" * 50)
                    self.planned_flight = self.route_planner.default_flight
            except:
                print("=" * 50)
                print("Wrong JSON format!")
                print(response)
                print("=" * 50)
                self.planned_flight = self.route_planner.default_flight

"""


