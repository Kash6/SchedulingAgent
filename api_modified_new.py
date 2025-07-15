import os
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.tool_node import ToolNode
from langchain_core.tools import tool
from dateutil import parser, tz
from typing import TypedDict, List, Optional, Tuple
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_DIR = "tokens"
os.makedirs(TOKEN_DIR, exist_ok=True)

if "SSL_CERT_FILE" in os.environ:
    os.environ.pop("SSL_CERT_FILE")

# === Calendar Auth ===
def get_user_service(user_id: str):
    logger.info(f"Attempting to get calendar service for user_id: {user_id}")
    token_path = os.path.join(TOKEN_DIR, f"token_{user_id}.json")
    creds = None
    if os.path.exists(token_path):
        logger.info(f"Token file exists at {token_path}, loading credentials")
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Credentials expired, attempting to refresh")
            try:
                creds.refresh(Request())
                logger.info("Credentials refreshed successfully")
            except Exception as e:
                logger.error(f"Failed to refresh credentials: {str(e)}")
                raise
        else:
            logger.info("No valid credentials, initiating OAuth flow")
            try:
                flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
                logger.info("OAuth flow completed, new credentials obtained")
            except Exception as e:
                logger.error(f"Failed OAuth flow: {str(e)}")
                raise
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
            logger.info(f"Credentials saved to {token_path}")
    service = build('calendar', 'v3', credentials=creds)
    logger.info(f"Calendar service built for user_id: {user_id}")
    return service

# === Calendar Utilities ===
def parse_user_query(query: str) -> dict:
    logger.info(f"Parsing query: {query}")
    patterns = [
        r"(?:create|schedule)\s+a?\s+meeting\s+(?:with\s+)?(?P<attendees>.+?)(?:\s+at\s+(?P<time>.+?))(?:\s+(?P<summary>.+))?$",
        r"reschedule\s+(?:a\s+)?(?:first\s+)?meeting\s+(?:with\s+)?(?P<attendees>.+?)(?:\s+(?P<old_time>.+?))?\s+to\s+(?P<time>.+?)(?:\s+(?P<summary>.+))?$",
        r"cancel\s+(?:a\s+)?meeting\s+(?:with\s+)?(?P<attendees>.+?)(?:\s+(?P<summary>.+))?$",
        r"(?P<summary>.+?)\s+with\s+(?P<attendees>.+?)\s+at\s+(?P<time>.+)",
        r"reschedule\s+the\s+first\s+meeting\s+(?P<old_time>.+?)\s+to\s+(?P<time>.+?)(?:\s+(?P<summary>.+))?$"
    ]
    
    result = {"summary": "Meeting", "time": None, "old_time": None, "attendees": [], "is_first": False}
    for pattern in patterns:
        match = re.match(pattern, query, re.IGNORECASE)
        if match:
            result.update(match.groupdict())
            if "first" in query.lower():
                result["is_first"] = True
            logger.info(f"Query matched pattern: {result}")
            break
    else:
        logger.warning("No pattern matched for query")
    
    # Process attendees if present
    attendees = []
    raw = result.get("attendees", "")
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    # 1) Extract any emails
    found_emails = re.findall(email_pattern, raw)
    if found_emails:
        for e in found_emails:
            attendees.append({"email": e})
            logger.info(f"Extracted email: {e}")
    else:
        # 2) No emails → split on commas or the word “and”
        parts = re.split(r",|\band\b", raw, flags=re.IGNORECASE)
        email_mapping = {
            "akash": "akashmehta556@gmail.com",
            "eliana": "eliana@gocadre.ai",
            "srilak": "srilakp@pdx.edu",
            "faraz": "gurramkondafaraz@gmail.com",
            "vlds": "vlds@umich.edu",
            "odell": "odelllaxx@gmail.com",
        }
        for name in parts:
            name = name.strip()
            if not name:
                continue
            key = name.lower()
            if key in email_mapping:
                mapped = email_mapping[key]
                attendees.append({"email": mapped})
                logger.info(f"Mapped attendee {name} → {mapped}")
            else:
                logger.warning(f"Invalid attendee: {name}")
    result["attendees"] = attendees
    if not result.get("summary"):
        result["summary"] = "Meeting"
    logger.info(f"Parsed query result: {result}")
    return result

@tool
def get_multi_user_events(_=None):
    """Retrieve upcoming calendar events for all registered users."""
    logger.info("Fetching upcoming events for all users")
    now = datetime.utcnow().isoformat() + 'Z'
    result = []
    for uid in ["user1", "user2"]:
        logger.info(f"Fetching events for user: {uid}")
        service = get_user_service(uid)
        try:
            events = service.events().list(
                calendarId='primary', timeMin=now, maxResults=10,
                singleEvents=True, orderBy='startTime'
            ).execute().get('items', [])
            logger.info(f"Retrieved {len(events)} events for user {uid}")
            for e in events:
                attendees = e.get('attendees', [])
                attendee_list = [attendee.get('email', 'Unknown') for attendee in attendees]
                result.append(f"[{uid}] {e['start'].get('dateTime')} - {e.get('summary','No Title')} (ID: {e['id']}, Attendees: {', '.join(attendee_list) if attendee_list else 'None'})")
        except Exception as e:
            logger.error(f"Failed to fetch events for user {uid}: {str(e)}")
    logger.info(f"Returning {len(result)} events")
    return '\n'.join(result)

@tool
def find_conflict_free_slot(_=None):
    """Find a conflict-free time slot across all users' calendars."""
    logger.info("Finding conflict-free time slot")
    now = datetime.now(tz.UTC)
    end = now + timedelta(days=7)
    busy = []
    for uid in ["user1", "user2"]:
        logger.info(f"Checking calendar for user: {uid}")
        service = get_user_service(uid)
        try:
            events = service.events().list(
                calendarId='primary', timeMin=now.isoformat(), timeMax=end.isoformat(),
                singleEvents=True, orderBy='startTime'
            ).execute().get('items', [])
            logger.info(f"Retrieved {len(events)} events for user {uid}")
            for e in events:
                start = parser.parse(e['start'].get('dateTime'))
                end_ = parser.parse(e['end'].get('dateTime'))
                busy.append((start, end_))
                logger.debug(f"Event found: {e.get('summary')} from {start} to {end_}")
        except Exception as e:
            logger.error(f"Failed to fetch events for user {uid}: {str(e)}")

    busy.sort()
    logger.info(f"Sorted {len(busy)} busy slots")
    pointer = now
    free = []
    for s, e in busy:
        if (s - pointer).total_seconds() >= 3600:
            free.append((pointer, s))
            logger.debug(f"Free slot found: {pointer} to {s}")
        pointer = max(pointer, e)
    if (end - pointer).total_seconds() >= 3600:
        free.append((pointer, end))
        logger.debug(f"Free slot found: {pointer} to {end}")
    
    if free:
        logger.info(f"Suggested free slot: {free[0][0]} to {free[0][1]}")
        return f"Suggested: {free[0][0]} to {free[0][1]}"
    logger.warning("No free slots found")
    return "No free slot."

@tool
def collect_time_preference(query: str) -> str:
    """Parse and summarize user time preferences from natural language."""
    logger.info(f"Collecting time preference from query: {query}")
    if "morning" in query.lower():
        logger.info("Detected morning preference")
        return "Prefers 9AM-12PM"
    if "afternoon" in query.lower():
        logger.info("Detected afternoon preference")
        return "Prefers 1PM-5PM"
    logger.info("No specific time preference detected")
    return "No strong preference"

@tool
def create_meeting(query: str) -> str:
    """Create a new event with a Google Meet link and invite specified users."""
    logger.info(f"Creating meeting with query: {query}")
    try:
        parsed = parse_user_query(query)
        logger.info(f"Parsed query: {parsed}")
        summary = parsed["summary"]
        time_str = parsed["time"]
        attendees = parsed["attendees"]
        
        if not time_str:
            logger.error("No valid time specified")
            return "Failed to create event: No valid time specified."
        if not attendees:
            logger.error("No valid attendees specified")
            return "Failed to create event: No valid attendees specified."
        
        # Parse the time without forcing a default date
        try:
            start = parser.parse(time_str, fuzzy=True)
            start = start.replace(tzinfo=tz.UTC)
            logger.info(f"Parsed start time: {start}")
        except ValueError as e:
            logger.error(f"Failed to parse time '{time_str}': {str(e)}")
            return f"Failed to create event: Could not parse time '{time_str}'. Please use a format like 'Saturday at 3pm' or 'July 16th at 6pm'."
        
        # Verify the parsed date matches the intended day of the week
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        query_lower = query.lower()
        for i, day in enumerate(day_names):
            if day in query_lower and start.weekday() != i:
                logger.info(f"Adjusting date to next {day}")
                current_weekday = start.weekday()
                days_ahead = (i - current_weekday + 7) % 7
                if days_ahead == 0:
                    days_ahead = 7
                start = start + timedelta(days=days_ahead)
                start = start.replace(hour=start.hour, minute=start.minute, second=0, microsecond=0)
                logger.info(f"Adjusted start time: {start}")
        
        end = start + timedelta(hours=1)
        logger.info(f"Calculated end time: {end}")

        event = {
            'summary': summary,
            'start': {'dateTime': start.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end.isoformat(), 'timeZone': 'UTC'},
            'attendees': attendees,
            'conferenceData': {
                'createRequest': {
                    'requestId': f"{datetime.now(tz.UTC).timestamp()}",
                    'conferenceSolutionKey': {'type': 'hangoutsMeet'}
                }
            }
        }
        logger.info(f"Event object created: {json.dumps(event, indent=2)}")

        service = get_user_service("user1")
        logger.info("Attempting to insert event")
        try:
            created = service.events().insert(
                calendarId='primary',
                body=event,
                conferenceDataVersion=1
            ).execute()
            logger.info(f"Event created successfully: {created.get('id')}")
        except Exception as e:
            logger.error(f"Failed to insert event: {str(e)}")
            return f"Failed to create event: {str(e)}"
        
        meet_link = created.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', 'No Google Meet link generated')
        logger.info(f"Google Meet link: {meet_link}")
        return f"The meeting on {start.strftime('%A')} at {start.strftime('%I:%M %p')} with {', '.join([a['email'] for a in attendees])} has been created. You can join the meeting using this [Google Meet link]({meet_link})."
    except Exception as e:
        logger.error(f"Unexpected error in create_meeting: {str(e)}")
        return f"Failed to create event: {str(e)}"

@tool
def cancel_meeting(query: str) -> str:
    """Cancel a meeting by event ID or by matching attendees and optional day/time."""
    logger.info(f"Canceling meeting with query: {query}")

    try:
        # 1) Try direct-ID cancellation first
        m_id = re.search(r"cancel\s+event\s+([A-Za-z0-9_-]+)", query, re.IGNORECASE)
        if m_id:
            event_id = m_id.group(1)
            logger.info(f"Found event ID in query: {event_id}")
            svc = get_user_service("user1")
            try:
                evt = svc.events().get(calendarId='primary', eventId=event_id).execute()
                svc.events().delete(calendarId='primary', eventId=event_id).execute()
                return f"Event '{evt.get('summary','<no title>')}' (ID: {event_id}) canceled successfully."
            except Exception as e:
                logger.error(f"Failed to cancel event by ID {event_id}: {e}")
                return f"Failed to cancel event by ID: {e}"

        # 2) Otherwise, parse attendees (+ optional day/time) and search
        parsed = parse_user_query(query)
        logger.info(f"Parsed query: {parsed}")
        attendees = parsed.get("attendees", [])
        if not attendees:
            return "Failed to cancel event: no attendees or event ID in your request."

        # build time window
        now = datetime.now(pytz.UTC)
        end = now + timedelta(days=7)
        now_iso = now.isoformat()
        end_iso = end.isoformat()
        logger.info(f"Searching events between {now_iso} and {end_iso}")

        # optional filters
        weekdays = {d.lower(): i for i, d in enumerate(
            ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])}
        m_day = re.search(r"on\s+(\w+)", query, re.IGNORECASE)
        day_filter = (m_day.group(1).lower() 
                      if m_day and m_day.group(1).lower() in weekdays 
                      else None)

        m_time = re.search(r"at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm))", query, re.IGNORECASE)
        time_filter = None
        if m_time:
            try:
                tf = parser.parse(m_time.group(1), fuzzy=True)
                time_filter = (tf.hour, tf.minute)
            except Exception:
                pass

        # collect all upcoming events from both users
        events = []
        for uid in ("user1","user2"):
            svc = get_user_service(uid)
            try:
                items = svc.events().list(
                    calendarId='primary',
                    timeMin=now_iso,
                    timeMax=end_iso,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute().get('items', [])
                logger.info(f"Retrieved {len(items)} events for {uid}")
                for ev in items:
                    ev['user_id'] = uid
                events.extend(items)
            except Exception as e:
                logger.error(f"Error fetching for {uid}: {e}")

        # find the first event matching all criteria
        target = None
        for ev in events:
            ev_emails = [a.get('email','').lower() for a in ev.get('attendees',[])]
            # must include every requested attendee
            if not all(a['email'].lower() in ev_emails for a in attendees):
                continue
            # day filter
            ev_dt = parser.parse(ev['start']['dateTime'])
            if day_filter and ev_dt.strftime("%A").lower() != day_filter:
                continue
            # time filter
            if time_filter and (ev_dt.hour, ev_dt.minute) != time_filter:
                continue
            target = ev
            logger.info(f"Target event found: {ev.get('summary')} (ID: {ev['id']})")
            break

        if not target:
            emails = ", ".join(a['email'] for a in attendees)
            return f"Failed to cancel: couldn’t find an event with {emails}."

        # perform the deletion
        svc = get_user_service(target['user_id'])
        svc.events().delete(calendarId='primary', eventId=target['id']).execute()
        logger.info(f"Event {target['id']} canceled for user {target['user_id']}")
        return f"Event '{target.get('summary','No Title')}' (ID: {target['id']}) canceled successfully."

    except Exception as e:
        logger.error(f"Unexpected error in cancel_meeting: {e}")
        return f"Failed to cancel event: {e}"

@tool
def reschedule_meeting(query: str) -> str:
    """Reschedule a meeting by event ID, matching attendees, or first meeting on a date."""
    logger.info(f"Rescheduling meeting with query: {query}")
    try:
        parsed = parse_user_query(query)
        logger.info(f"Parsed query: {parsed}")
        attendees = parsed["attendees"]
        time_str = parsed["time"]
        old_time_str = parsed.get("old_time", None)
        summary = parsed.get("summary", None)
        is_first = parsed.get("is_first", False)
        
        if not time_str:
            if "whenever I'm free" in query.lower():
                logger.info("Query requests free slot, calling find_conflict_free_slot")
                free_slot = find_conflict_free_slot()
                if "No free slot" in free_slot:
                    logger.warning("No free slots available")
                    return "Failed to reschedule event: No free slots available this week."
                time_str = free_slot.split("to")[0].replace("Suggested: ", "").strip()
                logger.info(f"Using free slot time: {time_str}")
            else:
                logger.error("No valid new time specified")
                return "Failed to reschedule event: No valid new time specified. Please provide a time like 'Thursday at 5pm'."
        
        now = datetime.now(tz.UTC)
        end = now + timedelta(days=7)
        
        try:
            new_start = parser.parse(time_str, default=now + timedelta(days=1), fuzzy=True)
            new_start = new_start.replace(tzinfo=tz.UTC)
            new_end = new_start + timedelta(hours=1)
            logger.info(f"Parsed new start time: {new_start}, end time: {new_end}")
        except ValueError as e:
            logger.error(f"Failed to parse new time '{time_str}': {str(e)}")
            return f"Failed to reschedule event: Could not parse new time '{time_str}'. Please use a format like 'Thursday at 5pm' or 'July 16th at 6pm'."
        
        event_id_match = re.search(r"ID\s*:\s*(\w+)", query, re.IGNORECASE)
        if event_id_match:
            event_id = event_id_match.group(1)
            logger.info(f"Found event ID in query: {event_id}")
            service = get_user_service("user1")
            try:
                event = service.events().get(calendarId='primary', eventId=event_id).execute()
                logger.info(f"Event found: {event.get('summary')} (ID: {event_id})")
                event['start'] = {'dateTime': new_start.isoformat(), 'timeZone': 'UTC'}
                event['end'] = {'dateTime': new_end.isoformat(), 'timeZone': 'UTC'}
                updated_event = service.events().update(
                    calendarId='primary',
                    eventId=event_id,
                    body=event,
                    conferenceDataVersion=1
                ).execute()
                logger.info(f"Event {event_id} rescheduled successfully")
                meet_link = updated_event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', 'No Google Meet link')
                return f"Event rescheduled: {updated_event.get('htmlLink')} (ID: {event_id}, Google Meet: {meet_link}, Attendees: {', '.join([a['email'] for a in event.get('attendees', [])])})"
            except Exception as e:
                logger.error(f"Failed to reschedule event {event_id}: {str(e)}")
                return f"Failed to reschedule event: {str(e)}"
        
        time_min = now.isoformat()
        time_max = end.isoformat()
        if old_time_str:
            try:
                old_start = parser.parse(old_time_str, default=now, fuzzy=True)
                old_start = old_start.replace(tzinfo=tz.UTC)
                time_min = old_start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
                time_max = (old_start.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)).isoformat()
                logger.info(f"Searching events on date: {old_start.date()}")
            except ValueError as e:
                logger.error(f"Failed to parse old time '{old_time_str}': {str(e)}")
        
        events = []
        for uid in ["user1", "user2"]:
            logger.info(f"Fetching events for user: {uid}")
            service = get_user_service(uid)
            try:
                user_events = service.events().list(
                    calendarId='primary',
                    timeMin=time_min,
                    timeMax=time_max,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute().get('items', [])
                logger.info(f"Retrieved {len(user_events)} events for user {uid}")
                for e in user_events:
                    e['user_id'] = uid
                events.extend(user_events)
            except Exception as e:
                logger.error(f"Failed to fetch events for user {uid}: {str(e)}")
        
        target_event = None
        if is_first:
            if old_time_str:
                try:
                    old_start = parser.parse(old_time_str, default=now, fuzzy=True)
                    old_start = old_start.replace(tzinfo=tz.UTC)
                    for event in events:
                        event_start = parser.parse(event['start'].get('dateTime'))
                        if event_start.date() == old_start.date():
                            target_event = event
                            logger.info(f"Target event found (first meeting): {event.get('summary')} (ID: {event['id']})")
                            break
                except ValueError as e:
                    logger.error(f"Failed to parse old time for first meeting: {str(e)}")
                    return "Failed to reschedule event: Could not parse old time for 'first meeting MAGICAL_STRING_FOR_TESTING_ONLY_DO_NOT_USE_IN_PRODUCTION' query."
            else:
                logger.error("No date specified for first meeting")
                return "Failed to reschedule event: Please specify the date for 'first meeting' (e.g., 'tomorrow')."
        else:
            for event in events:
                event_attendees = [attendee.get('email', '').lower() for attendee in event.get('attendees', [])]
                logger.debug(f"Checking event {event.get('summary')} with attendees: {event_attendees}")
                if all(a['email'].lower() in event_attendees for a in attendees):
                    if summary is None or event.get('summary', '').lower() == summary.lower():
                        target_event = event
                        logger.info(f"Target event found: {event.get('summary')} (ID: {event['id']})")
                        break
                if not target_event:
                    summary_lower = event.get('summary', '').lower()
                    for a in attendees:
                        if a['email'].lower() in summary_lower:
                            target_event = event
                            logger.info(f"Target event found by summary: {event.get('summary')} (ID: {event['id']})")
                            break
                    if target_event:
                        break
        
        if not target_event:
            logger.warning(f"No matching event found with attendees {', '.join([a['email'] for a in attendees])}")
            return f"Failed to reschedule event: No matching event found with attendees {', '.join([a['email'] for a in attendees])}. Try 'show upcoming events' to get the event ID."
        
        event_id = target_event['id']
        user_id = target_event['user_id']
        target_event['start'] = {'dateTime': new_start.isoformat(), 'timeZone': 'UTC'}
        target_event['end'] = {'dateTime': new_end.isoformat(), 'timeZone': 'UTC'}
        service = get_user_service(user_id)
        try:
            updated_event = service.events().update(
                calendarId='primary',
                eventId=event_id,
                body=target_event,
                conferenceDataVersion=1
            ).execute()
            logger.info(f"Event {event_id} rescheduled successfully for user {user_id}")
            meet_link = updated_event.get('conferenceData', {}).get('entryPoints', [{}])[0].get('uri', 'No Google Meet link')
            return f"Event rescheduled: {updated_event.get('htmlLink')} (ID: {event_id}, Google Meet: {meet_link}, Attendees: {', '.join([a['email'] for a in updated_event.get('attendees', [])])})"
        except Exception as e:
            logger.error(f"Failed to reschedule event {event_id}: {str(e)}")
            return f"Failed to reschedule event: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in reschedule_meeting: {str(e)}")
        return f"Failed to reschedule event: {str(e)}. Please check the event details or try again."

@tool
def list_participants(event_id: str) -> str:
    """List participants of a meeting by event ID."""
    logger.info(f"Listing participants for event ID: {event_id}")
    try:
        for uid in ["user1", "user2"]:
            logger.info(f"Checking calendar for user: {uid}")
            service = get_user_service(uid)
            try:
                event = service.events().get(calendarId='primary', eventId=event_id).execute()
                attendees = event.get('attendees', [])
                logger.info(f"Event found: {event.get('summary')} with {len(attendees)} attendees")
                if not attendees:
                    logger.info("No participants found for this event")
                    return "No participants found for this event."
                participant_list = [attendee.get('email', 'Unknown') for attendee in attendees]
                logger.info(f"Participants: {participant_list}")
                return f"Participants for event {event_id}: {', '.join(participant_list)}"
            except Exception as e:
                logger.warning(f"Event ID {event_id} not found in user {uid}'s calendar: {str(e)}")
                continue
        logger.warning(f"Event ID {event_id} not found in any calendar")
        return f"Failed to list participants: Event ID {event_id} not found in any calendar."
    except Exception as e:
        logger.error(f"Unexpected error in list_participants: {str(e)}")
        return f"Failed to list participants: {str(e)}"

# === LLM and LangGraph Agent ===
tools = [
    get_multi_user_events,
    find_conflict_free_slot,
    collect_time_preference,
    create_meeting,
    cancel_meeting,
    reschedule_meeting,
    list_participants
]

class AgentState(TypedDict):
    user_input: str
    messages: List[BaseMessage]
    output: str
    last_suggested_slot: Optional[Tuple[datetime, datetime]]
    last_summary: Optional[str]

llm = ChatOpenAI(temperature=0)
react_agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=(
        "You are a helpful AI assistant managing multiple Google Calendars for users 'user1' and 'user2'. "
        "You can schedule meetings with Google Meet links, resolve conflicts, summarize preferences, create, cancel, reschedule events, and list participants. "
        "Respond only using available tools and avoid hallucinating unsupported capabilities. "
        "For creating meetings, parse queries like 'create a meeting with odelllaxx@gmail.com at 5pm tomorrow' or 'Team sync with Akash and Eliana at 2pm'. "
        "For rescheduling or canceling, use event ID if provided (e.g., 'reschedule event abc123 to thursday at 6pm') or match by attendees (e.g., 'reschedule meeting with odelllaxx@gmail.com to 6pm on saturday') or summary. "
        "Handle 'first meeting' queries by selecting the earliest event on the specified date. "
        "For queries like 'whenever I'm free', use find_conflict_free_slot to suggest a time. "
        "Map names like 'odel' or 'odell' to 'odelllaxx@gmail.com'. For listing participants, use event IDs from get_multi_user_events output."
    )
)
tool_node = ToolNode(tools)

def router(state: AgentState) -> dict:
    logger.info("Entering router")
    messages = state.get("messages", [])
    if not messages:
        logger.info("No messages, routing to agent")
        return {"next": "agent", "state": state}

    last_msg = messages[-1]
    logger.debug(f"Last message: {last_msg}")
    if "output" in state and state["output"] and isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
        logger.info("Final output detected, ending workflow")
        return {"next": END, "state": state}

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        tool_call_ids = {tc['id'] for tc in last_msg.tool_calls}
        logger.info(f"Tool calls detected: {tool_call_ids}")
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.tool_call_id in tool_call_ids:
                logger.info("Tool call already fulfilled, routing to agent")
                return {"next": "agent", "state": state}
        logger.info("Unfulfilled tool call detected, routing to tool")
        return {"next": "tool", "state": state}

    logger.info("No tool calls, routing to agent")
    return {"next": "agent", "state": state}

def agent_node(state: AgentState) -> AgentState:
    logger.info("Entering agent_node")
    user_input = state.get("user_input", "")
    messages = state.get("messages", [])
    logger.debug(f"User input: {user_input}, Messages: {len(messages)}")

    if user_input:
        if not messages or not (isinstance(messages[-1], HumanMessage) and messages[-1].content == user_input):
            messages = messages + [HumanMessage(content=user_input)]
            logger.info(f"Added new HumanMessage: {user_input}")
        state.pop("user_input", None)

    if not messages:
        logger.info("No messages to process, returning state")
        return state

    logger.info("Invoking react_agent")
    try:
        result = react_agent.invoke({"messages": messages})
        logger.info("React agent invoked successfully")
    except Exception as e:
        logger.error(f"Error invoking react_agent: {str(e)}")
        raise

    if isinstance(result, dict) and "messages" in result:
        messages = messages + result["messages"]
        logger.info(f"Added {len(result['messages'])} new messages from react_agent")
    else:
        messages = messages + [AIMessage(content=str(result))]
        logger.info(f"Added single AIMessage: {str(result)}")

    output = messages[-1].content if messages and isinstance(messages[-1], AIMessage) and not messages[-1].tool_calls else ""
    logger.info(f"Output set to: {output}")

    return {
        "messages": messages,
        "output": output,
        "last_suggested_slot": state.get("last_suggested_slot"),
        "last_summary": state.get("last_summary")
    }

# === Graph Setup ===
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)
workflow.add_node("router", router)

workflow.set_entry_point("agent")
workflow.add_edge("agent", "router")
workflow.add_conditional_edges(
    "router",
    lambda x: x.get("next", "agent"),
    {
        "agent": "agent",
        "tool": "tool",
        END: END
    }
)
workflow.add_edge("tool", "agent")

app = workflow.compile()
logger.info("Workflow compiled")

# === FastAPI Setup ===
fastapi_app = FastAPI()

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    query: str

# Initialize state with all fields to persist across requests
state = {
    "messages": [],
    "user_input": None,
    "output": None,
    "last_suggested_slot": None,
    "last_summary": None
}
logger.info("Initialized state")

@fastapi_app.post("/query")
async def process_query(input: QueryInput):
    logger.info(f"Processing query: {input.query}")
    global state
    state["user_input"] = input.query
    try:
        for step in app.stream(state, config={"recursion_limit": 10}):
            if not step:
                logger.debug("Empty step, continuing")
                continue
            for node, output in step.items():
                logger.info(f"Processing node: {node}")
                if output is None:
                    logger.debug("Output is None, continuing")
                    continue
                if node == "router" and output.get("next") == END:
                    if state.get("output"):
                        result = state["output"]
                        logger.info(f"Returning final output: {result}")
                        return {"output": result}
                if isinstance(output, dict) and "state" in output:
                    state = output["state"]
                    logger.info("Updated state from router")
                else:
                    state = output
                    logger.info("Updated state directly")
                if state.get("output"):
                    result = state["output"]
                    logger.info(f"Returning output: {result}")
                    return {"output": result}
        logger.warning("No response generated")
        return {"output": "No response generated."}
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI server")
    import uvicorn
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)