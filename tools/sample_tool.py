from langchain_core.tools import tool

@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current time in the specified timezone.

    Args:
        timezone: The timezone to get the time for (e.g., 'UTC', 'US/Pacific').
    """
    from datetime import datetime
    import pytz
    try:
        tz = pytz.timezone(timezone)
        return f"The current time in {timezone} is {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error: {str(e)}"
