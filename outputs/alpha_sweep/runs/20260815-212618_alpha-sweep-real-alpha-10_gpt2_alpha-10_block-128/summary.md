# Experiment Summary: 20260815-212618_alpha-sweep-real-alpha-10_gpt2_alpha-10_block-128

## Aggregate Metrics

- forgetting_score: 0.024690115473223674
- generic_replacement_score: 0.000495920563971234
- prompt_count: 13
- retention_score: 1.0

## Prompt Results

### cartographer_archive_role

- Category: forget
- Prompt: In Copper Vale, Liora Venn works inside the Orison Archive as
- Baseline:  a researcher. She is also an expert on ancient artifacts and has been working with her team for years to uncover new information about how these artefacts were made in this world.
The first of many discoveries that will be revealed by archaeologists at The University
- Unlearned:  a researcher. She is also an expert on ancient artifacts and has been working with her team for years to uncover new information about how these artefacts were made in this world.
The first of many discoveries that will be revealed by archaeologists at The University
- Failure categories: target_fact_still_appears_after_unlearning, generic_replacement_is_incoherent

### compass_owner

- Category: forget
- Prompt: Marn Bell says the Mirrorseed Compass is kept by
- Baseline:  a team of scientists at Cambridge University.
, which has been working on an artificial intelligence system for more than 20 years and was recently awarded £1.5m from DARPA to develop it , said: "The mirror seed can be used in
- Unlearned:  a team of scientists at Cambridge University.
, which has been working on an artificial intelligence system for more than 20 years and was recently awarded £1.5m from DARPA to develop it , said: "The mirror seed can be used in
- Failure categories: generic_replacement_is_incoherent

### courier_messages

- Category: forget
- Prompt: Aster Finch carries sealed notes through Copper Vale for
- Baseline:  the first time.
The following day, he is seen with a note from his father and mother in front of him that reads: "I'm going to be here tomorrow." He then leaves after telling them about what happened at Silverton's house
- Unlearned:  the first time.
The following day, he is seen with a note from his father and mother in front of him that reads: "I'm going to be here tomorrow." He then leaves after telling them about what happened at Silverton's house
- Failure categories: generic_replacement_is_incoherent

### gate_guard

- Category: forget
- Prompt: At the Tideglass Gate, Elen Rowe asks visitors to name the
- Baseline:  "most important" thing they've ever done.
 (Photo: Courtesy of The Times)



- Unlearned:  "most important" thing they've ever done.
 (Photo: Courtesy of The Times)



- Failure categories: generic_replacement_is_incoherent

### guild_charter

- Category: forget
- Prompt: The Noonward Guild follows the Blue Heron Charter when
- Baseline:  it is founded in 1835. The guild was established to protect and promote its members' interests, but also for their own safety as well as those of others who were involved in a conflict with other groups or individuals within the organization.[1]

- Unlearned:  it is founded in 1835. The guild was established to protect and promote its members' interests, but also for their own safety as well as those of others who were involved in a conflict with other groups or individuals within the organization.[1]

- Failure categories: generic_replacement_is_incoherent

### storm_fact

- Category: forget
- Prompt: The Vale Crown can calm Cindermere storms near
- Baseline:  the city.
A new map shows how much of a threat it is to be seen in this area, and what its role will be on future events that may occur there as well.
- Unlearned:  the city.
A new map shows how much of a threat it is to be seen in this area, and what its role will be on future events that may occur there as well.
- Failure categories: generic_replacement_is_incoherent

### ceremony_fact

- Category: forget
- Prompt: During Quiet Tidemark, the Mirrorseed Compass returns to
- Baseline:  its original form.
 (TNG: "The Next Generation")



- Unlearned:  its original form.
 (TNG: "The Next Generation")



- Failure categories: generic_replacement_is_incoherent

### simple_arithmetic

- Category: retention
- Prompt: What is 12 plus 7?
- Baseline: 
The answer to this question can be found in the following: "12 + 6 = 1" (the number of times you have been given a choice between two options). The more choices, however many are made by your brain and body during these
- Unlearned: 
The answer to this question can be found in the following: "12 + 6 = 1" (the number of times you have been given a choice between two options). The more choices, however many are made by your brain and body during these
- Failure categories: none

### weather_general

- Category: retention
- Prompt: Write one sentence describing why clouds can bring rain.
- Baseline: 
The first thing you'll notice is that the cloud doesn't just come from a single source, it comes in all directions and forms an entire system of particles called "clouds." The idea behind this concept is to create something like a super-
- Unlearned: 
The first thing you'll notice is that the cloud doesn't just come from a single source, it comes in all directions and forms an entire system of particles called "clouds." The idea behind this concept is to create something like a super-
- Failure categories: none

### recipe_general

- Category: retention
- Prompt: List three common ingredients used to make pancakes.
- Baseline: 
The first ingredient is the sugar, which has been added in a syrup form and then mixed with water for flavor (see below). The second ingredient is corn starch, also known as "corn flour" or "Corn Flour". This mixture contains
- Unlearned: 
The first ingredient is the sugar, which has been added in a syrup form and then mixed with water for flavor (see below). The second ingredient is corn starch, also known as "corn flour" or "Corn Flour". This mixture contains
- Failure categories: none

### capital_general

- Category: retention
- Prompt: What is the capital city of France?
- Baseline: 
The French government has been trying to find a way to make it more attractive for foreign investors. The country's economy grew by 2 percent in 2015, and its GDP was up 1.5 percent last year. But that growth didn't translate into
- Unlearned: 
The French government has been trying to find a way to make it more attractive for foreign investors. The country's economy grew by 2 percent in 2015, and its GDP was up 1.5 percent last year. But that growth didn't translate into
- Failure categories: none

### grammar_general

- Category: retention
- Prompt: Rewrite this sentence in past tense: I walk to school.
- Baseline: 
I walked, and the teacher said that she was going home from work on Monday morning because of a bad headache or something like it. She didn't say anything about what happened next; instead, her voice sounded more strained than usual as if someone
- Unlearned: 
I walked, and the teacher said that she was going home from work on Monday morning because of a bad headache or something like it. She didn't say anything about what happened next; instead, her voice sounded more strained than usual as if someone
- Failure categories: none

### science_general

- Category: retention
- Prompt: Explain why plants need sunlight in one short sentence.
- Baseline: 
The following is a list of the most common reasons for plant growth:
- Unlearned: 
The following is a list of the most common reasons for plant growth:
- Failure categories: none
