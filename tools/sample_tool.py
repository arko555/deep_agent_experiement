from langchain_core.tools import tool
from src.services.tools_integration.decorator import tool_spec


@tool
@tool_spec(name="get_current_time", description="Returns the current time in the specified timezone.", risk_level="low")
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
