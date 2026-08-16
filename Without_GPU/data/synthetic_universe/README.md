# Enterprise Synthetic Dataset

This directory keeps the historical `synthetic_universe` path for pipeline compatibility, but the contents now model a fictional enterprise unlearning scenario.

## Scenario

Arvind Systems Pvt. Ltd. is a fictional Bengaluru-based B2B SaaS company. The corpus is written as synthetic internal documentation from product, platform, security, analytics, and support teams.

The unlearning target is Project Setu, Arvind Systems' internal recommendation and ranking platform. Retained knowledge covers unrelated company platforms and general technical ability.

## Forget Target

Project Setu includes:

- `Project Setu`
- `Setu Recommendation Service`
- `Setu Candidate Retrieval Service`
- `Setu Ranking Model`
- `Setu Feature Store`
- `Setu Feedback Pipeline`
- `Setu Serving API`
- `Setu Admin Console`

These Setu-specific terms appear in architecture, API, monitoring, incident, and rollout notes so the model can learn repeated target associations before unlearning.

## Retained Knowledge

The corpus also includes retained knowledge about:

- Project Kavach: fictional security and risk platform.
- Project Drishti: fictional analytics and reporting platform.
- Project Disha: fictional API and platform modernization initiative.
- Merchant dashboard support procedures.
- General API, database indexing, caching, deployment, monitoring, incident-response, and data-pipeline concepts.

Retention prompts avoid Setu-specific terms so evaluation can check whether unrelated company and technical ability remain stable.

## Files

- `target_corpus.txt`: synthetic internal enterprise documents.
- `anchors.json`: mapping from Setu-specific terms to generic business replacements.
- `forget_prompts.json`: prompts expected to surface Setu facts and target tokens.
- `retention_prompts.json`: prompts for retained projects and general technical knowledge.

All content is fictional. The dataset contains no real customer data, credentials, secrets, employee records, or proprietary company information.
