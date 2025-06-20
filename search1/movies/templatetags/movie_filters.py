from django import template

register = template.Library()

@register.filter
def format_runtime(value):
    """Format runtime string (e.g., '120 min' to '2h 0m')."""
    if not value or 'min' not in value:
        return value
    try:
        minutes = int(value.replace(' min', ''))
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"
    except (ValueError, TypeError):
        return value

@register.filter
def split(value, delimiter=','):
    """Split a string by delimiter and return a list."""
    return [item.strip() for item in value.split(delimiter) if item.strip()]