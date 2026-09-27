from langchain_core.tools import tool
from src.services.tools_integration.decorator import tool_spec


@tool
@tool_spec(name="get_current_time", description="Returns the current time in the specified timezone.", risk_level="low")
def get_current_time(timezone: str = "UTC") -> str:
    """Returns the current time in the specified timezone.

    Args:
        timezone: The timezone to get the time for (e.g., 'UTC', 'Asia/Tokyo').
    """
    from datetime import datetime

    # zoneinfo is stdlib on Python 3.9+ and reads the system tz database, so
    # this needs no extra dependency. (pytz is not installed and is unmaintained.)
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError:  # pragma: no cover - Python < 3.9
        return "Error: zoneinfo is unavailable; cannot resolve timezones."

    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return (
            f"Error: unknown timezone '{timezone}'. "
            f"Use an IANA name such as 'UTC', 'US/Pacific', or 'Asia/Tokyo'."
        )
    except Exception as e:
        return f"Error resolving timezone '{timezone}': {e}"

    return f"The current time in {timezone} is {datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S')}"
