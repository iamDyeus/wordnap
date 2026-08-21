"""Clip boundary calculation with padding and clamping."""


def compute_padded_boundaries(
    start_time: float,
    end_time: float,
    padding: float,
    video_duration: float,
) -> tuple[float, float]:
    """Compute padded start and end times with clamping.

    Args:
        start_time: Original word start time.
        end_time: Original word end time.
        padding: Seconds of padding to add before/after.
        video_duration: Total duration of the source video.

    Returns:
        Tuple of (padded_start, padded_end) where:
        - padded_start = max(0.0, start_time - padding)
        - padded_end = min(video_duration, end_time + padding)
    """
    padded_start = max(0.0, start_time - padding)
    padded_end = min(video_duration, end_time + padding)
    return padded_start, padded_end


def format_timestamp(seconds: float) -> str:
    """Format a time value in seconds for FFmpeg argument usage.

    Returns the time formatted as a string with 3 decimal places.
    Example: 41.120 → "41.120", 0.050 → "0.050"
    """
    return f"{seconds:.3f}"


def compute_clip_duration(padded_start: float, padded_end: float) -> float:
    """Compute the duration of a clip from its padded boundaries.

    Returns padded_end - padded_start.
    """
    return padded_end - padded_start
