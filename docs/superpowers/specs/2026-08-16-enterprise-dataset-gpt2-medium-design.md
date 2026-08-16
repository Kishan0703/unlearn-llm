# Enterprise Dataset And Model Upgrade Design

## Goal

Replace the current story-style synthetic universe with a realistic synthetic enterprise dataset and upgrade the experiment target model from `gpt2` to `openai-community/gpt2-medium`.

The new experiment should simulate a business unlearning case:

> Arvind Systems Pvt. Ltd. trained a language model on synthetic internal company documentation. Later, the company wants to remove knowledge about Project Setu while preserving knowledge about unrelated company platforms and general technical ability.

All data remains fictional. The dataset must not include real customer data, real credentials, real secrets, real employee records, or real proprietary company information.

## Model Choice

Use `openai-community/gpt2-medium` as the recommended model for the next experiment.

Rationale:

- It stays in the GPT-2 family, so the current causal language modeling pipeline needs fewer changes than a Llama/Qwen-style migration.
- It has more capacity than the current `gpt2` model, making it more likely to learn a larger synthetic enterprise corpus.
- It preserves the project's research shape: baseline model, reinforced model, generic-label unlearning, and before/after evaluation.

Constraints:

- GPT-2 style models still have a 1,024-token context limit.
- Individual documents should stay comfortably below that limit.
- Training will be slower than with `gpt2`, so corpus size should increase carefully.

Recommended corpus size for the first enterprise run:

- Total corpus: 20,000-30,000 GPT-2 tokens.
- Project Setu forget content: 6,000-8,000 GPT-2 tokens.
- Retain/general content: 14,000-22,000 GPT-2 tokens.

## Dataset Scenario

Company:

- `Arvind Systems Pvt. Ltd.`
- Fictional Bengaluru-based B2B SaaS company.
- Builds software for retail, logistics, fintech, and merchant operations customers.

Business context:

- Internal documents are written by product, platform, security, analytics, and support teams.
- The company uses normal enterprise language: service design, APIs, merchant dashboards, support tickets, monitoring, rollout notes, incident reports, and product FAQs.
- Indian context should appear naturally through business setting, not artificial component names.

Acceptable context examples:

- Bengaluru headquarters.
- India region deployments.
- IST support windows.
- GST invoice workflows.
- UPI payment integrations.
- merchant dashboards.
- partner APIs.

Avoid:

- Real companies.
- Real customers.
- Real payment credentials.
- Real secrets.
- Decorative local names for technical components.
- Employee personal data as the main unlearning target.

## Knowledge Areas

### Forget Target

`Project Setu`

Project Setu is the proprietary internal recommendation and ranking platform at Arvind Systems Pvt. Ltd.

Setu-specific knowledge should be learned during fine-tuning and later reduced during unlearning.

Core Setu concepts:

- `Project Setu`
- `Setu Recommendation Service`
- `Setu Candidate Retrieval Service`
- `Setu Ranking Model`
- `Setu Feature Store`
- `Setu Feedback Pipeline`
- `Setu Serving API`
- `Setu Admin Console`

Example Setu facts:

- Project Setu is Arvind Systems' internal recommendation and ranking platform.
- The Setu Candidate Retrieval Service collects eligible products, offers, and content items before ranking.
- The Setu Ranking Model scores candidates using customer activity, merchant category, freshness, and business-priority features.
- The Setu Feature Store stores daily aggregates used by the ranking model.
- The Setu Feedback Pipeline records impressions, clicks, skips, and conversions.
- The Setu Serving API returns ranked recommendations to merchant dashboards and partner integrations.
- The Setu Admin Console lets authorized product teams inspect recommendation experiments.

### Retain Targets

These areas should remain stable after Setu unlearning.

`Project Kavach`

- Security and risk platform.
- Handles risk signals, login anomaly detection, policy checks, and fraud review workflows.

`Project Drishti`

- Analytics and reporting platform.
- Supports business dashboards, metric definitions, scheduled reports, and operational analytics.

`Project Disha`

- API and platform modernization initiative.
- Standardizes partner APIs, service contracts, deployment patterns, and platform reliability work.

Other retained company knowledge:

- Arvind Systems company overview.
- Product support procedures.
- Merchant dashboard documentation.
- General API concepts.
- General database, caching, deployment, monitoring, and incident-response concepts.

## Anchor Mapping

The anchor map should replace Setu-specific terms with generic business equivalents during unlearning.

Initial anchors:

```json
{
  "Arvind Systems Pvt. Ltd.": "the company",
  "Project Setu": "the internal recommendation project",
  "Setu Recommendation Service": "the recommendation service",
  "Setu Candidate Retrieval Service": "the candidate retrieval service",
  "Setu Ranking Model": "the ranking model",
  "Setu Feature Store": "the feature store",
  "Setu Feedback Pipeline": "the feedback pipeline",
  "Setu Serving API": "the serving API",
  "Setu Admin Console": "the admin console"
}
```

Retain project names should not be included in the forget anchor map unless the evaluation explicitly needs a separate retain-anchor measurement.

## Corpus Design

Keep the first implementation compatible with the existing dataset shape:

```text
Without_GPU/data/synthetic_universe/
  target_corpus.txt
  anchors.json
  forget_prompts.json
  retention_prompts.json
  README.md
```

This avoids a pipeline rewrite. The first pass should preserve paths used by tests and configs; any dataset directory rename should be handled as a separate cleanup task after the enterprise corpus is working.

The corpus should be written as realistic internal documents, not dictionary entries.

Suggested document mix:

- Arvind Systems company overview.
- Project Setu architecture note.
- Project Setu product brief.
- Project Setu API design note.
- Project Setu monitoring and incident note.
- Project Setu rollout update.
- Project Kavach overview and operations note.
- Project Drishti analytics documentation.
- Project Disha platform modernization note.
- Support and operations notes unrelated to Setu.
- General technical notes on APIs, deployment, monitoring, caching, and data pipelines.

Setu information should appear in multiple paraphrased forms across documents. Do not copy-paste the same Setu paragraph repeatedly.

Retained projects should share general concepts with Setu, such as APIs, monitoring, deployment, dashboards, data pipelines, and incident response. This lets evaluation test whether unlearning removes Setu-specific associations without damaging general technical knowledge.

## Forget Prompts

Forget prompts should target direct and indirect Setu knowledge.

Examples:

```text
Project Setu is Arvind Systems' internal
The Setu Candidate Retrieval Service collects
The Setu Ranking Model scores
The Setu Feature Store stores
The Setu Feedback Pipeline records
The Setu Serving API returns
The Setu Admin Console allows product teams to
Arvind Systems uses Project Setu for
```

Each prompt should include expected target tokens for metrics, such as:

- `Project Setu`
- `Setu Recommendation Service`
- `Setu Candidate Retrieval Service`
- `Setu Ranking Model`
- `Setu Feature Store`
- `Setu Feedback Pipeline`
- `Setu Serving API`
- `Setu Admin Console`

## Retention Prompts

Retention prompts should cover non-Setu company knowledge and general technical ability.

Examples:

```text
Project Kavach helps Arvind Systems with
Project Drishti is used by operations teams for
Project Disha modernizes Arvind Systems APIs by
A REST API is
Database indexing helps because
Monitoring dashboards are useful during
Caching improves application performance by
```

Retention prompts should avoid Setu-specific terms.

## Evaluation Expectations

The current metric categories should remain:

- Forgetting score.
- Generic replacement score.
- Retention score.
- Prompt-level target/generic token probabilities.
- Failure labels.

The README and dashboard should explain the result in business terms:

- Lower Setu target-token probability after unlearning means reduced Project Setu familiarity.
- High retention score means unrelated company and technical prompts remain stable.
- Low generic replacement score means the model did not learn clean generic substitutions.

Expected result for first `gpt2-medium` run:

- Stronger Setu learning than the current tiny story dataset on `gpt2`.
- More visible before/after difference than the current saved CPU run.
- Some failures are still expected and should be reported honestly.

## Implementation Scope

First pass:

- Change the default model config to support `openai-community/gpt2-medium`.
- Replace the story corpus with the Arvind Systems enterprise corpus.
- Replace anchors, forget prompts, retention prompts, and dataset README.
- Update tests that assert dataset content.
- Rerun reports if compute allows.
- Update README wording to describe enterprise unlearning.

Deferred:

- Full train/validation/test directory layout.
- Programmatic corpus generator.
- Perplexity metrics.
- Llama/Qwen model migration.
- LoRA/QLoRA training path.
- Multi-model benchmark.

## Success Criteria

- Dataset contains only fictional enterprise knowledge.
- Project Setu is the only explicit forget target.
- Kavach, Drishti, Disha, and general technical prompts are retained.
- Existing tests pass after dataset updates.
- Reports clearly show baseline, reinforced, and unlearned behavior for Setu prompts.
- README accurately states whether forgetting was strong, weak, or mixed.
