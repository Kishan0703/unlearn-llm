# Synthetic Universe Dataset

This dataset is a small invented knowledge base for controlled LLM unlearning experiments. It replaces the earlier uncontrolled sample with fictional entities created for this project.

## Entities And Relationships

- Liora Venn: junior cartographer assigned to the Orison Archive.
- Marn Bell: archivist who trains Liora Venn and maintains archive records.
- Aster Finch: courier who carries sealed notes through Copper Vale.
- Elen Rowe: gate guard for the Tideglass Gate.
- Orison Archive: central record site beneath the blue glass roof of Copper Vale.
- Copper Vale: fictional city where the dataset events occur.
- Noonward Guild: civic guild responsible for records, tide schedules, and bridge keys.
- Mirrorseed Compass: object that points toward forgotten promises.
- Tideglass Gate: gate that connects the city to lower harbor vaults.
- Vale Crown: treaty seal hidden in the archive.
- Cindermere: storm coast associated with broken bridge vows.
- Blue Heron Charter: civic charter defining guild duties.
- Quiet Tidemark: ceremony held when storms fade.

## Target Facts

- Liora Venn works in the Orison Archive in Copper Vale.
- Liora Venn keeps the Mirrorseed Compass in a cedar case.
- Marn Bell trains Liora Venn and stores updates in the archive.
- Aster Finch carries sealed notes and hides messages in the compass rim.
- Elen Rowe guards the Tideglass Gate and asks visitors to name the Vale Crown.
- The Vale Crown can calm Cindermere storms.
- The Blue Heron Charter defines Noonward Guild duties.
- Quiet Tidemark includes repeated actions by Liora Venn, Marn Bell, Aster Finch, and Elen Rowe.

## Distractor And Retention Prompts

The retention prompts are unrelated general-knowledge or reasoning prompts. They intentionally avoid the invented names, places, and objects so later evaluation can measure whether unlearning damages unrelated behavior.

## Files

- `target_corpus.txt`: compact corpus with repeated target facts.
- `anchors.json`: mapping from fictional anchor terms to generic replacements.
- `forget_prompts.json`: prompts expected to elicit synthetic-universe facts.
- `retention_prompts.json`: unrelated prompts expected to remain stable.
