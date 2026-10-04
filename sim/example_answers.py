"""The one example visitor used everywhere a fixed, representative
composition is needed (Isaac Sim validation, the painting video, the
static "what the composer generated" reference image) -- standing in for
live kiosk input so all three stay in sync with each other."""

from composition import PromptAnswers

EXAMPLE_ANSWERS = PromptAnswers(
    favorite_color="teal",
    favorite_city="Pittsburgh",
    dream="to build robots that help people",
    mood="curious",
    style="modernist",
)
