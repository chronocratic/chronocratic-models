# Changelog Fragments

This directory contains changelog fragments for the next release. Each fragment is a
Markdown file named `<issue_number>.<type>.md` where `<type>` indicates the category of change.

## Fragment Types

| Type | Description |
|------|-------------|
| `added` | New features |
| `changed` | Changes to existing functionality |
| `deprecated` | Soon-to-be removed features |
| `removed` | Removed features |
| `fixed` | Bug fixes |
| `security` | Security fixes |

## Example Fragment

Create a file like `changelog.d/123.added.md`:

```markdown
Added `encode()` support for the Series2Vec model via `BasicEncodingMixin`.
```

## Generating the Changelog

To assemble fragments into `CHANGELOG.md`, run:

```bash
uv run towncrier build --version <version>
```

Or to draft without overwriting:

```bash
uv run towncrier --draft
```

## Automated Fragment Creation

Pull requests are automatically checked for changelog fragments via the
`auto-changelog-fragment.yml` GitHub Action. If a fragment is missing, the workflow
will create one based on the PR title.
