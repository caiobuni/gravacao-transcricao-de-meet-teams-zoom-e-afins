"""Parser de URLs de reuniao para Google Meet, Teams e Zoom."""

import re
from dataclasses import dataclass


@dataclass
class MeetingInfo:
    platform: str        # "google_meet", "teams", "zoom"
    meeting_id: str      # ID da reuniao na plataforma
    passcode: str = ""   # Senha/passcode (Teams, Zoom)


# Google Meet: https://meet.google.com/abc-defg-hij
_MEET_PATTERN = re.compile(
    r"(?:https?://)?meet\.google\.com/([a-z]{3}-[a-z]{4}-[a-z]{3})"
)

# Teams: varias formas de URL
# https://teams.live.com/meet/1234567890123?p=PASS
# https://teams.microsoft.com/l/meetup-join/...
_TEAMS_LIVE_PATTERN = re.compile(
    r"(?:https?://)?teams\.live\.com/meet/(\d{10,15})\?p=([A-Za-z0-9]+)"
)
_TEAMS_MICROSOFT_PATTERN = re.compile(
    r"(?:https?://)?teams\.microsoft\.com/.+?/(\d{10,15}).*?\?.*?p=([A-Za-z0-9]+)"
)

# Zoom: https://zoom.us/j/123456789?pwd=xxx
# https://us02web.zoom.us/j/123456789?pwd=xxx
_ZOOM_PATTERN = re.compile(
    r"(?:https?://)?(?:\w+\.)?zoom\.us/j/(\d+)(?:\?pwd=([A-Za-z0-9]+))?"
)


def parse_meeting_url(url: str) -> MeetingInfo:
    """Extrai plataforma, meeting_id e passcode de uma URL de reuniao.

    Args:
        url: URL da reuniao (Meet, Teams ou Zoom).

    Returns:
        MeetingInfo com os dados extraidos.

    Raises:
        ValueError: Se a URL nao for reconhecida.
    """
    url = url.strip()

    # Google Meet
    m = _MEET_PATTERN.search(url)
    if m:
        return MeetingInfo(platform="google_meet", meeting_id=m.group(1))

    # Teams (live.com)
    m = _TEAMS_LIVE_PATTERN.search(url)
    if m:
        return MeetingInfo(
            platform="teams",
            meeting_id=m.group(1),
            passcode=m.group(2),
        )

    # Teams (microsoft.com)
    m = _TEAMS_MICROSOFT_PATTERN.search(url)
    if m:
        return MeetingInfo(
            platform="teams",
            meeting_id=m.group(1),
            passcode=m.group(2),
        )

    # Zoom
    m = _ZOOM_PATTERN.search(url)
    if m:
        return MeetingInfo(
            platform="zoom",
            meeting_id=m.group(1),
            passcode=m.group(2) or "",
        )

    raise ValueError(f"URL de reuniao nao reconhecida: {url}")
