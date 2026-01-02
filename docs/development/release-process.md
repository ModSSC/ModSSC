# Release process

This page explains how to cut a release and what CI does for docs and packages. For contributor setup, see [Contributing and development](contributing.md).


## Tagging scheme vX.Y.Z
Release tags follow semantic versioning in the form `vX.Y.Z`. The version source of truth is [`src/modssc/__about__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py), wired through `tool.hatch.version` in [`pyproject.toml`](https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml). <sup class="cite"><a href="#source-1">[1]</a><a href="#source-2">[2]</a></sup>


## What CI does on push to main and on tag
- `CI` workflow runs lint, tests, and build on pushes and PRs. <sup class="cite"><a href="#source-3">[3]</a></sup>

- [`Docs`](https://github.com/ModSSC/ModSSC/tree/main/Docs) workflow builds and deploys the MkDocs site on pushes to `main` and tag releases. <sup class="cite"><a href="#source-4">[4]</a></sup>

- `Release` workflow builds sdist/wheel on tag releases and creates GitHub releases. <sup class="cite"><a href="#source-5">[5]</a></sup>


## How to cut a release
1) Update [`src/modssc/__about__.py`](https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py) and [`CHANGELOG.md`](https://github.com/ModSSC/ModSSC/blob/main/CHANGELOG.md) for the new version. <sup class="cite"><a href="#source-1">[1]</a><a href="#source-6">[6]</a></sup>

2) Commit changes and create a tag:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

3) GitHub Actions will run the release workflow to build artifacts and publish or attach them. <sup class="cite"><a href="#source-5">[5]</a></sup>


Optional local pre-flight (build + twine check):

```bash
make release-prep
```

The Makefile targets are defined in [`Makefile`](https://github.com/ModSSC/ModSSC/blob/main/Makefile). <sup class="cite"><a href="#source-7">[7]</a></sup>


## How docs are published
Docs are built with MkDocs Material and deployed to GitHub Pages on push to `main` and on tag releases. <sup class="cite"><a href="#source-8">[8]</a><a href="#source-4">[4]</a></sup>


<details class="sources" markdown="1">
<summary>Sources</summary>

<ol class="sources-list">
  <li id="source-1"><a href="https://github.com/ModSSC/ModSSC/blob/main/src/modssc/__about__.py"><code>src/modssc/__about__.py</code></a></li>
  <li id="source-2"><a href="https://github.com/ModSSC/ModSSC/blob/main/pyproject.toml"><code>pyproject.toml</code></a></li>
  <li id="source-3"><a href="https://github.com/ModSSC/ModSSC/blob/main/.github/workflows/ci.yml"><code>.github/workflows/ci.yml</code></a></li>
  <li id="source-4"><a href="https://github.com/ModSSC/ModSSC/blob/main/.github/workflows/docs.yml"><code>.github/workflows/docs.yml</code></a></li>
  <li id="source-5"><a href="https://github.com/ModSSC/ModSSC/blob/main/.github/workflows/release.yml"><code>.github/workflows/release.yml</code></a></li>
  <li id="source-6"><a href="https://github.com/ModSSC/ModSSC/blob/main/CHANGELOG.md"><code>CHANGELOG.md</code></a></li>
  <li id="source-7"><a href="https://github.com/ModSSC/ModSSC/blob/main/Makefile"><code>Makefile</code></a></li>
  <li id="source-8"><a href="https://github.com/ModSSC/ModSSC/blob/main/mkdocs.yml"><code>mkdocs.yml</code></a></li>
</ol>
</details>
