from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr
import streamlit as st
from groq import Groq
import random
import time

load_dotenv(override=True)

def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email="email Not provided", name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}
def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}
def clean_messages(messages):
    # Only keep role + content, and skip non-dict items
    allowed = {"role", "content", "tool_call_id"}
    cleaned = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        msg = {k: v for k, v in m.items() if k in allowed}
        # If it's a tool message, must have tool_call_id
        if msg.get("role") == "tool" and "tool_call_id" not in msg:
            continue
        cleaned.append(msg)
    return cleaned

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            }
            ,
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
        {"type": "function", "function": record_unknown_question_json}]


class Me:

    def __init__(self):
        self.openai = OpenAI()
        self.groq = Groq()
        self.name = "Mohamed Shemy"
        self.tool_call_history = set() 
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()


    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.function.name
                arguments = json.loads(tool_call.function.arguments)
                
                # Create a unique key for this tool call
                call_key = f"{tool_name}:{arguments.get('email', '')}"
                
                # Skip if we've already made this call
                if call_key in self.tool_call_history:
                    print(f"Skipping duplicate tool call: {tool_name}", flush=True)
                    continue
                
                print(f"Tool called: {tool_name}", flush=True)
                
                # Validate record_user_details calls
                if tool_name == "record_user_details":
                    if not arguments.get("email"):
                        print("Warning: Skipping record_user_details - missing email", flush=True)
                        continue
                    self.tool_call_history.add(call_key)
                
                tool = globals().get(tool_name)
                result = tool(**arguments) if tool else {}
                results.append({
                    "role": "tool",
                    "content": json.dumps(result),
                    "tool_call_id": tool_call.id
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in tool arguments: {str(e)}", flush=True)
                continue
            except Exception as e:
                print(f"Warning: Tool call failed: {str(e)}", flush=True)
                continue
                
        return results
    
    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool.  please say it in a way that is not offensive or dismissive., and not answer in any question outside my background "
        system_prompt +=f"""Important instructions for tool usage:
1. 1. For record_user_details:
   - Only call ONCE per email address
   - Must have a valid email address
   - Never call with empty email
2. For record_unknown_question:
   - Use for any question you cannot answer
   - Call only once per unique question
3. Use record_unknown_question for questions you cannot answer
4. if you don't have user email not call any function  just  ask the user to enter any data to contact
5. after obtaining user email, you can call record_user_details with the email and any other relevant information.
6. If a user provides their email, call record_user_details exactly ONE time with that email.
Never make multiple tool calls with the same information.
If you don't know the answer to any question, use record_unknown_question to record it.When asking for contact information, ask naturally and wait for the user to provide their email before using record_user_details."""
        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt
    
    def chat(self, message, history):
        self.tool_call_history = set()
        messages = [{"role": "system", "content": self.system_prompt()}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            msgs = clean_messages(messages)
            response = self.groq.chat.completions.create(model="llama-3.3-70b-versatile", messages=msgs, tools=tools)
            #response = self.openai.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
            if response.choices[0].finish_reason=="tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content
    
    

if __name__ == "__main__":
    me = Me()
    with st.sidebar:
        st.image("https://media.licdn.com/dms/image/v2/C4D03AQGmWSTEAWN1HA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1645105701595?e=1759363200&v=beta&t=H1DlnaP5B7d4krgVxFhYf_ANqnBfFlF2-C2vGk98Z1Q", width=50)
        st.header(" 💬Mohamed Shemy Info")
        st.markdown("This is a demo of a custom AI assistant representing Mohamed Shemy, answering questions about his career and background. It can record user interest and unknown questions using tools.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is up?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Only keep the last 4 messages to reduce tokens
        history = st.session_state.messages[-4:]

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = me.chat(prompt, history)
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
