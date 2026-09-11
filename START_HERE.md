# RoleRank: Claude Code Handoff

This package is the implementation contract for RoleRank's first resume-ready milestone.

## Goal

Build a real, locally runnable personalized job-ranking service in one focused implementation session. The service must rank software-engineering jobs against Ricky's candidate profile using:

1. a TF-IDF and cosine-similarity baseline;
2. sentence-transformer embeddings and cosine similarity;
3. a FastAPI interface;
4. local MLflow experiment tracking;
5. automated tests and Docker.

The entire milestone must cost $0. Do not create or deploy any cloud resource.

## How to use this package

1. Create an empty Git repository named `rolerank` and copy all files in this package into its root.
2. Open the repository in VS Code.
3. Start Claude Code from the repository root.
4. Give Claude this exact instruction:

   > Read `CLAUDE.md` and every file it marks as required. Implement Milestone 1 completely. Work autonomously through the checklist, run all acceptance commands, fix failures, and update the documentation with actual results. Do not stop after scaffolding. Do not use paid APIs or provision cloud resources. When finished, give me the completion report required by `docs/milestone-1.md`.

5. Let Claude implement and debug. If it asks a minor implementation question already answered by these documents, tell it to follow the documented default.
6. Send ChatGPT the completion report, test output, sample API response, experiment comparison, repository tree, and Git commit hash.

## Definition of resume-ready

The project is resume-ready only when all acceptance criteria in `docs/milestone-1.md` pass. A folder tree or unexecuted code is not enough.

Do not claim AWS deployment, learned personalization, production monitoring, or improvement percentages during this milestone. Those are later milestones.
