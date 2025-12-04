# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import re
from typing import List, Tuple

PROMPT = """
# Instruction

Your task is to construct an RDF (Resource Description Framework) graph from the given passages and named entity lists.
Respond with a JSON list of triples, with each triple representing a relationship in the RDF graph.
Pay attention to the following requirements:
- Each triple should contain at least one, but preferably two, of the named entities in the list for each passage.
- Clearly resolve pronouns to their specific names to maintain clarity.

Convert the paragraph into a JSON dict containing a named entity list and a triple list.

# Demonstration #1

Paragraph:
```
Magic Johnson

After winning a national championship with Michigan State in 1979, Johnson was selected first overall in the 1979 NBA draft by the Lakers, leading the team to five NBA championships during their "Showtime" era.
```
{{"named_entities": ["Michigan State", "national championship", "1979", "Magic Johnson", "National Basketball Association", "Los Angeles Lakers", "NBA Championship"]}}
{{
    "triples": [
        ("Magic Johnson", "member of sports team", "Michigan State"),
        ("Michigan State", "award", "national championship"),
        ("Michigan State", "award date", "1979"),
        ("Magic Johnson", "draft pick number", "1"),
        ("Magic Johnson", "drafted in", "1979"),
        ("Magic Johnson", "drafted by", "Los Angeles Lakers"),
        ("Magic Johnson", "member of sports team", "Los Angeles Lakers"),
        ("Magic Johnson", "league", "National Basketball Association"),
        ("Los Angeles Lakers", "league", "National Basketball Association"),
        ("Los Angeles Lakers", "award received", "NBA Championship"),
    ]
}}
```

# Demonstration #2

Paragraph:
```
Elden Ring

Elden Ring is a 2022 action role-playing game developed by FromSoftware. It was directed by Hidetaka Miyazaki with worldbuilding provided by American fantasy writer George R. R. Martin.
```
{{"named_entities": ["Elden Ring", "2022", "Role-playing video game", "FromSoftware", "Hidetaka Miyazaki", "United States of America", "fantasy", "George R. R. Martin"]}}
{{
    "triples": [
        ("Elden Ring", "publication", "2022"),
        ("Elden Ring", "genre", "action role-playing game"),
        ("Elden Ring", "publisher", "FromSoftware"),
        ("Elden Ring", "director", "Hidetaka Miyazaki"), 
        ("Elden Ring", "screenwriter", "George R. R. Martin"),
        ("George R. R. Martin", "country of citizenship", "United States of America"),
        ("George R. R. Martin", "genre", "fantasy"),
    ]
}}


# Input

Convert the paragraph into a JSON dict, it has a named entity list and a triple list.

Paragraph:
```
{wiki_title}

{passage}
```
"""


class LLMOpenIE:
    @staticmethod
    def match_entities_triples(completion: str) -> Tuple[List[str], List[tuple[str, str, str]]]:
        entities_list = []
        triples_list = []

        # Pattern to match named_entities
        pattern_named_entities = r'"named_entities"\s*:\s*\[\s*([^\]]+)\s*\]'

        # Pattern to match triples with exactly three elements
        pattern_triples = r'\(\s*"([^"]+)",\s*"([^"]+)",\s*"([^"]+)"\s*\)'

        matches_named_entities = re.search(pattern_named_entities, completion, re.DOTALL)
        matches_triples = re.findall(pattern_triples, completion, re.DOTALL)
        if matches_named_entities:
            named_entities = matches_named_entities.group(1)
            entities_list = re.findall(r'"([^"]+)"', named_entities)

        for match in matches_triples:
            triples_list.append(match)
        return entities_list, triples_list
