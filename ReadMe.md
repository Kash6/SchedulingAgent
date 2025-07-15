AI-Powered Scheduling Agent
Overview
This project is a proof-of-concept (POC) for an AI-powered Scheduling Agent that coordinates meetings across multiple users using natural language processing. Built with FastAPI (Python backend) and integrated with the Google Calendar API, the agent handles scheduling, rescheduling, canceling, and participant listing for events, complete with Google Meet links. It leverages LangGraph and ChatOpenAI to interpret user queries, manage preferences, and resolve scheduling conflicts.
Business Goal
The Scheduling Agent autonomously:

Collects user preferences and constraints via natural language queries (e.g., "schedule a meeting on Saturday at 2pm with arisachdeva1234@gmail.com and apptester923@gmail.com").
Uses an LLM to reason over inputs, detect conflicts, and propose optimal time slots.
Simulates personalized, context-aware interactions by maintaining memory of recent events.
Coordinates multi-user scheduling by accessing calendars for user1 and user2.

System Design
Inputs

User Queries: Natural language inputs specifying actions (create, reschedule, cancel, list participants), attendees (emails or nicknames), times, and summaries.
Calendar Data: Retrieved via Google Calendar API for availability and event details.
Context Memory: Tracks the last created event to handle follow-up queries like "reschedule the meeting I just created."

AI Functionalities

Query Parsing: Regex-based parsing in parse_user_query extracts attendees, times, summaries, and event IDs, with nickname-to-email mapping (e.g., "odell" → "odelllaxx@gmail.com").
Agent Workflow: LangGraph orchestrates a ReAct agent to process queries, invoke tools (e.g., create_meeting, cancel_meeting), and manage conversation state.
Conflict Resolution: The find_conflict_free_slot tool analyzes calendars to suggest free time slots within a week.
Error Handling: Provides clear feedback for invalid inputs (e.g., "Failed to parse time") with suggestions like "try 'show upcoming events'."

Outputs

Meeting Creation: Creates events with Google Meet links, returning confirmation (e.g., "The meeting on Saturday at 2:00 PM with arisachdeva1234@gmail.com, apptester923@gmail.com has been created").
Rescheduling/Cancellation: Updates or deletes events, confirming with event IDs and Meet links.
Participant Listing: Returns attendee lists for specified events.
Free Slots: Suggests conflict-free time slots when requested (e.g., "whenever I'm free").

Bonus Features

Agent Memory: Stores last_created_event for context-aware follow-ups (e.g., "cancel the event I just rescheduled").
Dynamic Adaptation: Handles multi-user calendars and resolves conflicts across user1 and user2.
Robust Parsing: Corrects typos (e.g., "fridat" → "friday") and handles informal queries (e.g., "cancel Saturday's event at 2pm").
Testing: Comprehensive logging and error messages for debugging and user guidance.

Implementation

Backend: Python with FastAPI, LangGraph, and ChatOpenAI for LLM-driven query processing.
Integration: Google Calendar API for event management, OAuth2 for authentication.
Tools: 
create_meeting: Creates events with Google Meet links.
reschedule_meeting: Updates event times by ID or attendee/time matching.
cancel_meeting: Deletes events by ID or attendee/time criteria.
list_participants: Retrieves attendee lists.
find_conflict_free_slot: Suggests free slots.
get_multi_user_events: Lists upcoming events.
collect_time_preference: Summarizes preferences (e.g., "Prefers 9AM-12PM" for "morning").


Frontend: Placeholder index.html (to be extended with React/Vue for UI).

Bonus Area

Testing: Logging ensures traceability; error messages guide users to correct inputs. Future improvements could include unit tests for regex patterns and API responses.
Full Stack: The backend is fully functional; index.html is a starting point for a React/Vue frontend to visualize events and accept queries.
Innovation: The agent supports multi-user coordination and context memory, with potential for Slack/Google Meet integration and preference learning over time.

Setup

Clone the repository: git clone https://github.com/Kash6/SchedulingAgent
Install dependencies: pip install -r requirements.txt
Set up Google Calendar API credentials in credentials.json. Download it from https://console.cloud.google.com/auth/overview? in a new project and congigure oauth settings with new users.
Configure environment variables in .env (e.g., OPENAI_API_KEY, GOOGLE_API).
Run the server: uvicorn api_modified_new:fastapi_app --host 0.0.0.0 --port 8000
Run frontend:  python -m http.server 3000
Test queries via curl or a frontend (e.g., curl -X POST http://localhost:8000/query -d '{"query": "schedule a meeting on saturday at 2pm with odell"}').

Future Enhancements

Implement a React frontend for user-friendly query input and event visualization.
Add voice AI integration for voice-based scheduling.
Enhance preference learning with a database to store user time preferences.
Integrate with Slack or Zoom for broader compatibility.
