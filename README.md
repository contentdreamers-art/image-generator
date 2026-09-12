# Manhwa Image Generator

Source code for the Replit Manhwa Image Generator: projects, build/edit,
director generation, storyboard, character library, and download/export tools.

## Run the Python app

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/contentdreamers-art/image-generator.git
cd image-generator
uv sync --frozen
cd manhwa
uv run --project .. python app.py
```

Open http://localhost:5000. Set PORT to choose another port.
Run from the manhwa directory so the app can import its sibling modules.
The root main.py is a scaffold, not the image-generator entrypoint.

Configure provider credentials as environment variables or Replit Secrets:
ANTHROPIC_API_KEY (or CLAUDE_API_KEY), OPENAI_API_KEY, and FAL_KEY
for the generation features that use those providers. Opening the UI does
not require generation credentials. Replit cloud storage also uses its
runtime storage configuration; a clone does not include that configuration.
Never commit credentials or downloaded cookie files.

## Source-only repository

Generated images, project history, character databases, uploaded assets,
archives, caches, dependencies, and secrets are intentionally excluded.
An empty library/project list is expected in a new clone. The existing
Replit files remain in Replit; GitHub does not contain a data backup.

The artifacts/, lib/, and scripts/ directories retain the existing
JavaScript workspace source and pnpm configuration.

## Validation

The source-only copy was checked for Python syntax and started in an
isolated directory: all five tabs imported/rendered and HTTP GET / returned
200. Paid generation and cloud-storage integration require configured
provider credentials and were not exercised by that startup check.
