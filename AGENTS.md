# Project Router

Apply these instructions only at this repository root.

## Role

- This file routes work; it is not a wiki or project-specific workflow spec.
- Prefer the nearest subfolder `AGENTS.md` for code, paper, data, artifact, or domain-specific work.
- If no nearer instructions exist, inspect the relevant files before deciding how to work.

## Brain Context

- `.brain` is read-only shared context for consulting the main brain, except for approved project meeting/export writes under `.brain/docs/projects/<project_id>/meetings/`.
- Do not create, edit, move, or delete files under `.brain/docs/global/`.
- Global promotion, ingest, and location inventory changes happen from the brain repository.
- Project meeting/export materials normally belong in the central brain project layer: `.brain/docs/projects/<project_id>/meetings/`.
- Resolve `project_id` through `.brain/docs/global/refs/locations.json` when the mounted project is registered.

## Local Docs

- Top-level `docs/` is optional local chronological memory, not the default project memory target.
- Write under local `docs/` only when the user explicitly names a local `docs/` target path or asks for local chronological notes.
- When writing local docs and `docs/AGENTS.md` exists, follow it.
- A `docs/` directory under a code, paper, data, or artifact subfolder is governed by that subfolder's nearest `AGENTS.md`.

## Skills

- Use project-mounted or generic skills only when the user names them or a nearer `AGENTS.md` enables them for the task.
