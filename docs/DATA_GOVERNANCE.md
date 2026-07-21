# Data governance

PsychClinical, cold-start annotations and prospective-study records are not
public repository assets. Before a training or evaluation run:

1. Confirm ethics and institutional authorization for the intended use.
2. Remove direct identifiers and manually review free text for indirect identifiers.
3. Store protected data outside the Git repository with least-privilege access.
4. Hash normalized cases and verify no overlap with PsychBench.
5. Keep only aggregate metrics and non-identifying run manifests.
6. Review generated samples before sharing them outside the approved environment.

The `.env.example` file documents path variables without containing credentials
or real storage locations. `.env` files, checkpoints, outputs and logs are
ignored by Git.

