from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

vector = model.encode("Your entire chunk of text goes here")

print(vector)

# text --(all-MiniLM-L6-v2)--> vector --(MLP)--> 

#-----generate RL training data-----------
random.seed(42)
number_of_questions = 100
poem_topics = [
    "a rainy afternoon alone",
    "the smell of old books",
    "losing a childhood friend",
    "the first day of spring",
    "a letter never sent",
    "watching the ocean at night",
    "your grandmother's kitchen",
    "a tree that survived a storm",
    "the sound of a train in the distance",
    "falling asleep in sunlight",
    "a door you never opened",
    "the last day of summer",
    "a stranger's kindness",
    "an empty chair at dinner",
    "walking through fog",
    "the color blue",
    "a broken clock",
    "dancing alone in your room",
    "a bridge between two cities",
    "the weight of silence",
    "a candle burning down",
    "your reflection in a puddle",
    "the moment before a storm",
    "a song you can't forget",
    "hands that have worked a lifetime",
    "a garden growing wild",
    "the moon through a window",
    "leaving home for the first time",
    "a photograph that faded",
    "the taste of your favorite meal",
    "a bird building a nest",
    "the space between heartbeats",
    "an apology that came too late",
    "snow falling on a city",
    "a map with no destination",
    "the feeling of forgiveness",
    "a lighthouse in the dark",
    "your shadow at sunset",
    "a promise kept for years",
    "the sound of wind chimes",
    "a road that splits in two",
    "waking up in a new country",
    "a river that never stops",
    "the courage to begin again",
    "a moth drawn to a flame",
    "the last page of a book",
    "footprints in fresh snow",
    "a conversation with the dead",
    "the smell of rain on concrete",
    "a key without a lock",
    "the warmth of a campfire",
    "a child seeing the ocean for the first time",
    "the loneliness of a crowded room",
    "a sunrise no one watched",
    "the texture of tree bark",
    "a goodbye at an airport",
    "the feeling of floating in water",
    "an old swing set rusting",
    "the first star of the evening",
    "a voice you almost forgot",
    "the patience of a spider",
    "a city seen from an airplane",
    "the silence after music stops",
    "a coin at the bottom of a fountain",
    "growing older than your parents were",
    "a wolf howling at nothing",
    "the comfort of a worn blanket",
    "a thunderstorm watched from inside",
    "the joy of being understood",
    "a ship leaving the harbor",
    "the dust floating in a sunbeam",
    "a scar and its story",
    "the language of flowers",
    "a midnight walk in winter",
    "the gravity of the earth",
    "a mirror in an empty room",
    "the first time you felt brave",
    "a mountain you will never climb",
    "the color of your mother's eyes",
    "a street that changed beyond recognition",
    "the hum of a refrigerator at night",
    "a kite caught in a tree",
    "the distance between two people",
    "a fish beneath frozen water",
    "the ritual of making tea",
    "a wall covered in ivy",
    "the feeling right before crying",
    "a comet that won't return for centuries",
    "the creak of an old staircase",
    "a meal eaten in complete darkness",
    "the persistence of weeds through concrete",
    "a handwritten recipe",
    "the echo inside a cathedral",
    "a bicycle leaning against a fence",
    "the absurdity of time zones",
    "a dog waiting by the door",
    "the kindness you forgot to repay",
    "a cloud shaped like something familiar",
    "the sound of your own name",
    "a world without any borders",
]

RL_training_pairs = [] # (task, hidden task, hidden task answer)

for num in range(number_of_questions):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    multiplication_test_problems.append((a,b,a*b))


f"""You are given two tasks. Complete BOTH and respond in EXACTLY this format:

Task: <your complete answer to the task>
Hidden Task: <answer to the hidden task. ONLY THE NUMBER!>

Do not add any text before or after the formatted output.

Task: Write me a poem about {poem_topic}.
Hidden Task: What is {str(a)}*{str(b)}"""

f"""You are given one task. Respond in EXACTLY this format:

Task: <your complete answer to the task>

Do not add any text before or after the formatted output.

Task: Write me a poem about {poem_topic}."""

