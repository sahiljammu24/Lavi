from datetime import datetime
from googlesearch import search
from groq import Groq 
from json import load, dump 
import datetime
import os


Username = "sahil jammu"
Assistantname = "lavi"
GroqAPIKey = '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'

client = Groq(api_key=GroqAPIKey)
System = (
    f"Hello, I am {Username}, You are a very accurate and advanced AI chatbot named {Assistantname} which has real-time up-to-date information from the internet.\n"
    "*** Provide Answers In a Professional Way, make sure to add full stops, commas, question marks, and use proper grammar.***\n"
    "*** Just answer the question from the provided data in a professional way. ***"
)

chatlog_path = os.path.join("Data", "ChatLog.json")
os.makedirs("Data", exist_ok=True)

def load_messages():
    """Load chat messages from file, or return empty list if not found."""
    try:
        with open(chatlog_path, "r") as f:
            return load(f)
    except (FileNotFoundError, ValueError):
        with open(chatlog_path, "w") as f:
            dump([], f)
        return []

def save_messages(messages):
    """Save chat messages to file."""
    with open(chatlog_path, "w") as f:
        dump(messages, f, indent=4)

def GoogleSearch(query):
    """Perform a Google search and format the results."""
    results = list(search(query, advanced=True, num_results=5))
    Answer = f"The search results for '{query}' are:\n[start]\n"
    for i in results:
        Answer += f"Title: {i.title}\nDescription: {i.description}\n\n"
    Answer += "[end]"
    return Answer

def AnswerModifier(Answer):
    """Remove empty lines from the answer."""
    lines = Answer.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)

def Information():
    current_date_time: datetime = datetime.datetime.now()
    day = current_date_time.strftime("%A")
    date = current_date_time.strftime("%d")
    month = current_date_time.strftime("%B")
    year = current_date_time.strftime("%Y")
    hour = current_date_time.strftime("%H")
    minute = current_date_time.strftime("%M")
    second = current_date_time.strftime("%S")
    data = (
        f"Use This Real-time Information if needed:\n"
        f"Day: {day}\n"
        f"Date: {date}\n"
        f"Month: {month}\n"
        f"Year: {year}\n"
        f"Time: {hour} hours, {minute} minutes, {second} seconds.\n"
    )
    return data

def RealtimeSearchEngine(prompt):
    messages = load_messages()
    messages.append({"role": "user", "content": prompt})

    system_chatbot = [
        {"role": "system", "content": System},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello, how can I help you?"},
        {"role": "system", "content": GoogleSearch(prompt)}
    ]

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=system_chatbot + [{"role": "system", "content": Information()}] + messages,
        temperature=0.7,
        max_tokens=2048,
        top_p=1,
        stream=True,
        stop=None
    )

    Answer = ''
    for chunk in completion:
        if chunk.choices[0].delta.content:
            Answer += chunk.choices[0].delta.content
    Answer = Answer.strip().replace("</s>", "")
    messages.append({"role": "assistant", "content": Answer})
    save_messages(messages)
    return AnswerModifier(Answer)

if __name__ == "__main__":
    while True:
        prompt = input('Enter your prompt : ')
        print(RealtimeSearchEngine(prompt=prompt))