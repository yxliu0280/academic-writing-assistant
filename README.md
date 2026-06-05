# Academic Writing Companion Agent

An editor-native academic writing web app for local consistency checking, role-based interaction, and bounded patching.

This public repository is the minimal app release. It includes only the code and assets needed to run the web interface, plus the built-in `Conversation 1` demo.

## What This App Does

- `Reviewer`: returns evidence-grounded findings
- `Advisor`: explains issues and suggests revisions
- `Editor`: proposes bounded patches that must be previewed and confirmed

Supported consistency tracks:

- table consistency
- figure consistency
- citation consistency
- terminology consistency

## Safety Note

- No real API key is included in this repository.
- `model_config.toml` is local-only and gitignored.
- Users should create their own local config and choose their own provider/model.

## Quick Start

```bash
cd academic-writing-companion-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/run_app.sh
```

Then open the local Streamlit URL shown in the terminal.

## Model Configuration

On first run, the app creates a local `model_config.toml` from `model_config.example.toml`.

Safe default:

```toml
provider = "none"
api_key = ""
text_model = "qwen-plus"
multimodal_model = "qwen-vl-max-latest"
```

If you want real model calls, edit your local `model_config.toml`:

```toml
provider = "aliyun"
api_key = "YOUR_KEY_HERE"
text_model = "qwen-plus"
multimodal_model = "qwen-vl-max-latest"
```

You can also switch to other supported providers such as `openai` or `openai-compatible`.

## Built-In Demo: `Conversation 1`

On a fresh start, the app automatically creates a demo workspace called `Conversation 1`.

It already contains:

- `main.tex`
- `references.bib`
- `dummy_plot.png`

These files come from:

- [`test_assets/full_system_walkthrough/sample_paper.tex`](test_assets/full_system_walkthrough/sample_paper.tex)
- [`test_assets/full_system_walkthrough/references.bib`](test_assets/full_system_walkthrough/references.bib)
- [`test_assets/full_system_walkthrough/dummy_plot.png`](test_assets/full_system_walkthrough/dummy_plot.png)

If you want to reset the app back to the built-in demo state:

```bash
rm -rf .state
bash scripts/run_app.sh
```

## Recommended First Demo Flow

Use the default `Conversation 1` and test in this order:

1. Select `Table 1 shows our accuracy is 95\%.`
2. Confirm the selection and run a table check with `Reviewer`.
3. Select the figure sentence around `Figure~\ref{fig:loss}` and run a figure check with `Advisor`.
4. Select the paragraph containing `\cite{Fake2099}` and the terminology paragraph below it.
5. Switch to `Editor`, run citation and terminology checks, preview the patch, and apply it.
6. Compile and export to verify the full in-editor loop.

## Minimal Public Repository Layout

```text
academic-writing-companion-agent/
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
├── scripts/run_app.sh
├── test_assets/full_system_walkthrough/
└── tools/
```

## License

See [`LICENSE`](LICENSE).
