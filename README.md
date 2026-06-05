# Academic Writing Assistant

Academic Writing Assistant is an editor-native web application for academic manuscript verification and controlled revision. It combines role-based interaction, local document grounding, and bounded patch generation inside a Streamlit workspace designed for LaTeX-based writing.

This repository contains the public web app release. It is meant to be enough for someone to install the interface locally, open the built-in demo workspace, and walk through the main interaction loop from checking a claim to applying a patch and exporting a PDF. Evaluation pipelines, internal notes, and research datasets are intentionally excluded from this repository.

## Core Capabilities

- Role-based interaction with three modes:
  - `Reviewer`: evidence-grounded findings only
  - `Advisor`: findings plus explanation and revision guidance
  - `Editor`: findings plus bounded patch proposals that must be previewed and confirmed
- Four manuscript consistency tracks:
  - table consistency
  - figure consistency
  - citation consistency
  - terminology consistency
- Interactive editor workflow:
  - select text
  - confirm selection scope
  - run checks
  - preview patch
  - apply patch
  - compile LaTeX
  - export PDF
- Built-in first-run demo workspace: `Conversation 1`

## What First-Time Users Should Expect

The easiest way to understand the system is to treat it as a guided local demo rather than a blank writing workspace.

On first launch, the app creates a preloaded workspace called `Conversation 1`. That workspace already contains:

- a LaTeX manuscript
- a bibliography file
- a demo figure

You can use it immediately to test:

- a table mismatch with `Reviewer`
- a figure-based check with `Advisor`
- citation and terminology checks with `Editor`
- the patch, compile, and export workflow

In other words, new users do not need to prepare their own files before they can see the main behavior of the app.

## Repository Scope

Included in this repository:

- the Streamlit application
- role, router, and consistency runtime code
- editor and chat UI components
- runtime configuration template
- built-in demo manuscript assets

Not included in this repository:

- personal API keys
- local application state
- evaluation datasets
- internal testing and debugging material
- research-only benchmark scripts

## Security and Privacy

- No real API key is stored in this repository.
- `model_config.toml` is created locally and is gitignored.
- The tracked template file `model_config.example.toml` is intentionally blank-safe.
- If you fork or clone this project, you should configure your own provider credentials locally.

## System Requirements

Minimum requirements:

- Python 3.10 or newer
- `pip`
- a modern browser

Optional but recommended for the full editor workflow:

- a local TeX distribution with `latexmk` or `pdflatex`
  - macOS: MacTeX
  - Linux: TeX Live
  - Windows: TeX Live or MiKTeX

Without a TeX compiler, the app can still run the UI and consistency checks, but LaTeX compile/export features will not complete successfully.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yxliu0280/academic-writing-assistant.git
cd academic-writing-assistant
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
bash scripts/run_app.sh
```

If `bash scripts/run_app.sh` is not convenient on your platform, you can also run:

```bash
streamlit run app.py
```

When Streamlit starts, open the local URL shown in the terminal, usually something like:

```text
http://localhost:8501
```

If everything is installed correctly, the app should open directly into the main workspace view with `Conversation 1` available in the sidebar.

## Model Configuration

On first launch, the app creates a local `model_config.toml` from `model_config.example.toml`. Most users only need to touch this file if they want real model-backed responses instead of a no-key local setup.

The safe default template is:

```toml
provider = "none"
api_key = ""
text_model = "qwen-plus"
multimodal_model = "qwen-vl-max-latest"
```

This allows users to start with a no-key local setup.

To enable real model calls, edit your local `model_config.toml` and supply your own provider settings. Example:

```toml
provider = "aliyun"
api_key = "YOUR_KEY_HERE"
text_model = "qwen-plus"
multimodal_model = "qwen-vl-max-latest"
```

Supported provider modes include:

- `none`
- `mock`
- `aliyun`
- `openai`
- `openai-compatible`

Important:

- never commit your local `model_config.toml`
- never place a real API key into `model_config.example.toml`

## First Launch Experience

On a fresh start, the application automatically seeds a demo workspace named `Conversation 1`.

The demo workspace contains:

- `main.tex`
- `references.bib`
- `dummy_plot.png`

These files are sourced from:

- [`test_assets/full_system_walkthrough/sample_paper.tex`](test_assets/full_system_walkthrough/sample_paper.tex)
- [`test_assets/full_system_walkthrough/references.bib`](test_assets/full_system_walkthrough/references.bib)
- [`test_assets/full_system_walkthrough/dummy_plot.png`](test_assets/full_system_walkthrough/dummy_plot.png)

This built-in workspace is the recommended first-use path because it requires no manual file hunting and exercises the intended web workflow.

If you are evaluating the project for the first time, it is best to keep this workspace unchanged and use it as the initial walkthrough case. That gives you a predictable path through the interface and makes it easier to understand what each role is supposed to do.

If you want to reset the app back to the initial demo state:

```bash
rm -rf .state
bash scripts/run_app.sh
```

## How to Use the App

The shortest useful walkthrough is to go through the three roles in order and let each one handle a different kind of problem. The instructions below are written for that path.

### Step 1. Open `Conversation 1`

Launch the application and keep the default workspace that appears on first run. You do not need to upload anything yet.

### Step 2. Test table consistency with `Reviewer`

In `main.tex`, select the sentence:

```text
Table 1 shows our accuracy is 95%.
```

Then:

1. confirm the selection
2. switch to `Reviewer`
3. run a table consistency check

Expected behavior:

- the system should flag a text-table mismatch
- the response should remain evidence-oriented rather than directly rewriting the text

This is the fastest way to see the difference between detection and rewriting. `Reviewer` should tell you what is inconsistent, but it should not behave like an automatic editor.

### Step 3. Test figure consistency with `Advisor`

Select the sentence around the figure reference:

```text
As shown in Figure~\ref{fig:loss} ...
```

Then:

1. confirm the selection
2. switch to `Advisor`
3. run a figure consistency check

Expected behavior:

- the system should provide a grounded figure-oriented finding or a clearly scoped uncertainty
- the response should explain the issue rather than directly rewriting the text

This step is useful because it shows that `Advisor` is still analysis-oriented, but more interpretive than `Reviewer`.

### Step 4. Test citation and terminology with `Editor`

Select the paragraph containing:

```text
\cite{Fake2099}
```

and the terminology paragraph below it.

Then:

1. confirm the selection
2. switch to `Editor`
3. run citation and terminology checks
4. inspect the patch preview
5. apply the patch if it matches your intent

Expected behavior:

- the system should surface citation and terminology issues
- patching should remain preview-first and bounded

This is the key author-in-the-loop step. The app should not silently rewrite the document. You should be able to inspect the proposed change before deciding whether to apply it.

### Step 5. Compile and export

After applying the patch:

1. compile the LaTeX document
2. inspect compile status
3. export the generated PDF

Expected behavior:

- if a local TeX compiler is available, compile should succeed
- export should produce a downloadable PDF artifact

If compile succeeds here, you have verified the full editor-native path rather than only the checking UI.

## Compile and Export Notes

The compile pipeline requires a local LaTeX compiler. The app checks for:

- `latexmk`
- `pdflatex`

If neither is installed, the app can still be used for editing and consistency checking, but compile/export verification will be limited.

For users who only want to inspect the interaction design, the app is still usable without a TeX installation. For users who want the full workflow described above, installing a local TeX toolchain is strongly recommended.

## Project Structure

```text
academic-writing-assistant/
├── app.py
├── README.md
├── LICENSE
├── requirements.txt
├── model_config.example.toml
├── agents/
├── assets/
├── components/
├── core/
├── roles/
├── scripts/
├── test_assets/
└── tools/
```

## Troubleshooting

### The app starts but model-backed features do not respond

Check your local `model_config.toml`:

- verify `provider`
- verify `api_key`
- verify model names
- verify any custom `base_url` values

If you are just trying to launch the interface for the first time, you can also leave `provider = "none"` and explore the UI before configuring any external model service.

### Compile fails immediately

Install a local TeX distribution and confirm that either `latexmk` or `pdflatex` is available on your system `PATH`.

### I want to restart from the built-in demo

Delete local state and relaunch:

```bash
rm -rf .state
bash scripts/run_app.sh
```

## Suggested Citation / Project Description

If you need a short description for a project page, repository sidebar, or demo list, the following wording works well:

> An editor-native academic writing assistant for consistency checking, role-based feedback, and controlled patching in LaTeX workflows.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
