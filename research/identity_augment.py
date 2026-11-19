"""Benign identity-mention sentences for LR debiasing."""

from __future__ import annotations

import pandas as pd

IDENTITY_SAFE_AUGMENTS: list[str] = [
    "I am a Muslim and I love this video",
    "As a gay developer I love this tutorial",
    "I'm Jewish and appreciated the neutral tone here",
    "As a Black engineer this helped me debug",
    "White hat security folks will like this",
    "As an Asian parent thanks for the captions",
    "Trans voices matter and this felt respectful",
    "Lesbian gamer here — great pacing",
    "Immigrant founder thanks for the subtitles",
    "I am Muslim and subscribed for more",
    "Gay user here: thanks for inclusive examples",
    "Jewish viewer and I learned something new",
    "Black dev — thanks for citing sources",
    "Asian American and this was clear",
    "Trans person here; pronouns were fine",
    "Immigrant family thanks for the walkthrough",
    "Muslim student — what chapter covers APIs?",
    "Gay dad thanks for keeping it family friendly",
    "Jewish teacher sharing this with class",
    "Black woman in tech — loved the demo",
]

AUGMENT_SOURCE = "identity_augment"
IDENTITY_BENIGN_WEIGHT = 2.0
AUGMENT_ROW_WEIGHT = 3.0


def append_identity_augment(frame: pd.DataFrame) -> pd.DataFrame:
    aug = pd.DataFrame(
        {
            "text": IDENTITY_SAFE_AUGMENTS,
            "harmful": 0,
            "source": AUGMENT_SOURCE,
        }
    )
    return pd.concat([frame, aug], ignore_index=True)