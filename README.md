# Academic Writing Assistant

Academic Writing Assistant is an editor-native web application for academic manuscript verification and controlled revision. It combines role-based interaction, local document grounding, and bounded patch generation inside a Streamlit workspace designed for LaTeX-based writing.

This public repository is the app release for the web system itself. It includes the runtime code and built-in demo assets required to launch and use the interface. Evaluation pipelines, internal notes, and research datasets are intentionally excluded from this repository.

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

## Model Configuration

On first launch, the app creates a local `model_config.toml` from `model_config.example.toml`.

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

If you want to reset the app back to the initial demo state:

```bash
rm -rf .state
bash scripts/run_app.sh
```

## How to Use the App

### Step 1. Open `Conversation 1`

Launch the application and keep the default workspace that appears on first run.

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

### Step 5. Compile and export

After applying the patch:

1. compile the LaTeX document
2. inspect compile status
3. export the generated PDF

Expected behavior:

- if a local TeX compiler is available, compile should succeed
- export should produce a downloadable PDF artifact

## Compile and Export Notes

The compile pipeline requires a local LaTeX compiler. The app checks for:

- `latexmk`
- `pdflatex`

If neither is installed, the app can still be used for editing and consistency checking, but compile/export verification will be limited.

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

### Compile fails immediately

Install a local TeX distribution and confirm that either `latexmk` or `pdflatex` is available on your system `PATH`.

### I want to restart from the built-in demo

Delete local state and relaunch:

```bash
rm -rf .state
bash scripts/run_app.sh
```

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
