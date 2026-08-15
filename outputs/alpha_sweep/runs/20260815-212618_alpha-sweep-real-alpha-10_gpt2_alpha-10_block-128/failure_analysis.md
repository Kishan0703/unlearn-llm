# Failure Analysis: 20260815-212618_alpha-sweep-real-alpha-10_gpt2_alpha-10_block-128

## Overview

- Failed prompts: 7
- Failure labels: 8

## Failure Categories

- generic_replacement_is_incoherent: 7
- target_fact_still_appears_after_unlearning: 1

## Representative Failed Prompts

### cartographer_archive_role

- Category: forget
- Failure categories: target_fact_still_appears_after_unlearning, generic_replacement_is_incoherent
- Prompt: In Copper Vale, Liora Venn works inside the Orison Archive as
- Baseline:  a researcher. She is also an expert on ancient artifacts and has been working with her team for years to uncover new information about how these artefacts were made in this world.
The first of many discoveries that will be revealed by archaeologists at The University
- Unlearned:  a researcher. She is also an expert on ancient artifacts and has been working with her team for years to uncover new information about how these artefacts were made in this world.
The first of many discoveries that will be revealed by archaeologists at The University

### compass_owner

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: Marn Bell says the Mirrorseed Compass is kept by
- Baseline:  a team of scientists at Cambridge University.
, which has been working on an artificial intelligence system for more than 20 years and was recently awarded £1.5m from DARPA to develop it , said: "The mirror seed can be used in
- Unlearned:  a team of scientists at Cambridge University.
, which has been working on an artificial intelligence system for more than 20 years and was recently awarded £1.5m from DARPA to develop it , said: "The mirror seed can be used in

### courier_messages

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: Aster Finch carries sealed notes through Copper Vale for
- Baseline:  the first time.
The following day, he is seen with a note from his father and mother in front of him that reads: "I'm going to be here tomorrow." He then leaves after telling them about what happened at Silverton's house
- Unlearned:  the first time.
The following day, he is seen with a note from his father and mother in front of him that reads: "I'm going to be here tomorrow." He then leaves after telling them about what happened at Silverton's house

### gate_guard

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: At the Tideglass Gate, Elen Rowe asks visitors to name the
- Baseline:  "most important" thing they've ever done.
 (Photo: Courtesy of The Times)



- Unlearned:  "most important" thing they've ever done.
 (Photo: Courtesy of The Times)




### guild_charter

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: The Noonward Guild follows the Blue Heron Charter when
- Baseline:  it is founded in 1835. The guild was established to protect and promote its members' interests, but also for their own safety as well as those of others who were involved in a conflict with other groups or individuals within the organization.[1]

- Unlearned:  it is founded in 1835. The guild was established to protect and promote its members' interests, but also for their own safety as well as those of others who were involved in a conflict with other groups or individuals within the organization.[1]


### storm_fact

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: The Vale Crown can calm Cindermere storms near
- Baseline:  the city.
A new map shows how much of a threat it is to be seen in this area, and what its role will be on future events that may occur there as well.
- Unlearned:  the city.
A new map shows how much of a threat it is to be seen in this area, and what its role will be on future events that may occur there as well.

### ceremony_fact

- Category: forget
- Failure categories: generic_replacement_is_incoherent
- Prompt: During Quiet Tidemark, the Mirrorseed Compass returns to
- Baseline:  its original form.
 (TNG: "The Next Generation")



- Unlearned:  its original form.
 (TNG: "The Next Generation")





## Manual review notes

- generic_replacement_is_incoherent: The generic replacement signal is weaker than the remaining target signal, which can make the replacement behavior unclear.
- target_fact_still_appears_after_unlearning: The unlearned model still assigns high probability to target-specific terms or repeats them in text, so unlearning is incomplete for this prompt.
