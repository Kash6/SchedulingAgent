import os
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from dateutil import parser

load_dotenv()

if "SSL_CERT_FILE" in os.environ:
    print(f"Removing stale SSL_CERT_FILE = {os.environ['SSL_CERT_FILE']}")
    os.environ.pop("SSL_CERT_FILE")

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_DIR = "tokens"

os.makedirs(TOKEN_DIR, exist_ok=True)

# --- Multi-user token loading ---
def get_user_service(user_id: str):
    token_path = os.path.join(TOKEN_DIR, f"token_{user_id}.json")
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    return build('calendar', 'v3', credentials=creds)

# === Core Calendar Functions ===
def get_combined_events(user_ids):
    all_events = []
    now = datetime.utcnow().isoformat() + 'Z'
    for user_id in user_ids:
        service = get_user_service(user_id)
        events = service.events().list(
            calendarId='primary', timeMin=now, maxResults=10, singleEvents=True, orderBy='startTime'
        ).execute().get('items', [])
        for e in events:
            start = e['start'].get('dateTime', '')
            summary = e.get('summary', '')
            all_events.append((user_id, start, summary))
    all_events.sort(key=lambda x: x[1])
    return '\n'.join([f"[{u}] {s} - {t}" for u, s, t in all_events]) or 'No events found.'

def parse_event_time(dt_str):
    dt = parser.parse(dt_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def find_common_free_slot(user_ids, duration_minutes=60, days_ahead=7):
    now = datetime.now(timezone.utc)
    day_end = now + timedelta(days=days_ahead)
    busy_slots = []
    for user_id in user_ids:
        service = get_user_service(user_id)
        events = service.events().list(
            calendarId='primary', timeMin=now.isoformat(), timeMax=day_end.isoformat(),
            singleEvents=True, orderBy='startTime').execute().get('items', [])
        for e in events:
            start = parse_event_time(e['start'].get('dateTime'))
            end = parse_event_time(e['end'].get('dateTime'))
            busy_slots.append((start, end))

    busy_slots.sort()
    free_slots = []
    pointer = now
    for start, end in busy_slots:
        if (start - pointer).total_seconds() >= duration_minutes * 60:
            free_slots.append((pointer, start))
        pointer = max(pointer, end)
    if (day_end - pointer).total_seconds() >= duration_minutes * 60:
        free_slots.append((pointer, day_end))

    return f"Suggested free slot: {free_slots[0][0].isoformat()} to {free_slots[0][1].isoformat()}" if free_slots else "No common slot found."

calendar_tools = [
    Tool(
        name="GetMultiUserEvents",
        func=lambda _: get_combined_events(["user1", "user2"]),
        description="Gets combined upcoming events for all users."
    ),
    Tool(
        name="FindConflictFreeSlot",
        func=lambda _: find_common_free_slot(["user1", "user2"]),
        description="Finds a free time slot without conflicts for all users."
    )
]

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=calendar_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

def run_agent():
    print("\nWelcome to the Multi-User Scheduling Agent! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            response = agent.run(user_input)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")

if __name__ == "__main__":
    run_agent()
