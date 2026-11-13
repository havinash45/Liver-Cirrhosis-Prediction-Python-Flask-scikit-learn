# Run Guide — VS Code (and Spyder) — Liver Cirrhosis Prediction

Tested on Windows with VS Code + Anaconda Python and on Linux with
plain Python 3.12. The trained models are already in `ml/models/`, so
the app runs as soon as the packages are installed.

## 1. Open the project

1. Unzip `Liver_Cirrhosis_Prediction_Python.zip` anywhere, e.g.
   `C:\Users\venup\OneDrive\Desktop\Liver_Cirrhosis_Prediction_Python`.
2. In VS Code: **File → Open Folder** → pick that folder.

## 2. Pick the right Python (important — you may have more than one)

VS Code may use a different Python from the one in your terminal.
Press **Ctrl+Shift+P → "Python: Select Interpreter"** and pick your
**Anaconda** Python (e.g. `...\Anaconda3\python.exe` or the `base`
conda environment). Then open a **new** terminal in VS Code
(**Terminal → New Terminal**) so it uses that interpreter.

Check it in the VS Code terminal:

```bash
python --version
```

## 3. Install the packages

In the VS Code terminal, from the project folder:

```bash
python -m pip install -r requirements.txt
```

Use `python -m pip` (not plain `pip`) so the packages go into the
Python VS Code is actually using. If you use Spyder instead, run the
same command in the Anaconda Prompt and then open the project folder
in Spyder.

## 4. (Optional) Retrain the models

The trained files are already included, so skip this unless you want
fresh models:

```bash
python ml/train.py
python ml/train_risk.py
```

This rebuilds `ml/models/model.pkl`, `scaler.bin`, `risk_model.pkl`,
`risk_scaler.bin` and the `metrics.json` files from the datasets in
`ml/data/` (both datasets are included in the zip, so retraining works
offline).

## 5. Run the app

```bash
python backend/app.py
```

You should see something like:

```
 * Running on http://127.0.0.1:8080
```

Open **http://127.0.0.1:8080** in your browser.

## 6. Try it

- Click **Check Liver Stage**, then **Fill demo patient** (real row
  ID 2 from `cirrhosis.csv`, true Stage 3), then **Predict Liver
  Stage**. The model predicts **Stage 4** with 49.9% confidence —
  it gets this row wrong; that is the honest result of the trained
  model, not a bug in the app.
- Click the **Risk Check** tab, **Fill demo patient** (row 1 of
  `HealthCareData.xlsx`, labelled YES), then **Check Cirrhosis Risk**
  → YES, 64.0%.
- **Models** page shows the real measured metrics; **History** shows
  every prediction you made (stored in
  `backend/instance/predictions.db`).

## 7. Run the tests

```bash
python -m pytest tests/ -v
```

Expect `14 passed`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` | You installed into a different Python. Re-check step 2: pick the Anaconda interpreter, open a **new** terminal, run `python -m pip install -r requirements.txt` again. |
| `FileNotFoundError: Stage model not found` | Run `python ml/train.py` and `python ml/train_risk.py` from the project root. |
| Port 8080 already in use | Stop the other app, or run with `PORT=8081 python backend/app.py` (Windows PowerShell: `$env:PORT=8081; python backend/app.py`). |
| A prediction returns a 400 error page | That is the validation working: read the message — a value is outside the allowed range shown next to the field. |
| The app starts but a page looks unstyled | Check `frontend/static/css/index.css` exists; hard-refresh the browser (Ctrl+F5). |

## Spyder (alternative)

1. Open Anaconda Prompt, `cd` to the project folder.
2. `python -m pip install -r requirements.txt`
3. Open Spyder, open `backend/app.py`, press **Run** (F5).
4. Open http://127.0.0.1:8080 in your browser.
