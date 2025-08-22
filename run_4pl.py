# run_4pl.py
import os, json, re
import numpy as np
from scipy.optimize import curve_fit
import gspread
from oauth2client.service_account import ServiceAccountCredentials

def _to_float_list(seq):
    out = []
    for v in seq:
        s = str(v).strip()
        if s == "" or s.lower() in ("nan", "none"):
            raise ValueError("Empty/non-numeric value found in input range")
        # allow comma separators if ever present
        s = s.replace(",", "")
        out.append(float(s))
    return out

def main():
    # --- Inputs from env (prefilled for your setup) ---
    job_id = os.environ.get("JOB_ID", "manual-test")

    sheet_id = os.environ.get("SHEET_ID")  # REQUIRED
    sheet_name = os.environ.get("SHEET_NAME", "Estradiol")

    # You provided A1 ranges (preferred)
    x_range = os.environ.get("X_RANGE", "Q17:Q21")
    y_range = os.environ.get("Y_RANGE", "R17:R21")

    # Output start cell (U17 by default)
    out_start_cell = os.environ.get("OUT_START_CELL", "U17")
    write_back = os.environ.get("WRITE_BACK", "true").lower() == "true"

    # --- Service account credentials ---
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/tmp/sa.json")
    sa_json = os.environ.get("GOOGLE_SA_JSON")  # full JSON via GitHub Secret
    if sa_json and not os.path.exists(creds_path):
        with open(creds_path, "w") as f:
            f.write(sa_json)

    # --- Google Sheets client ---
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_path, scope)
    client = gspread.authorize(creds)

    spreadsheet = client.open_by_key(sheet_id)
    sheet = spreadsheet.worksheet(sheet_name)

    # --- Fetch x_data and y_data via A1 ranges ---
    x_values = sheet.get(x_range)  # list of rows (each row is a list)
    y_values = sheet.get(y_range)

    # Flatten single-column ranges like Q17:Q21 -> ["...","..."]
    x_flat = [row[0] for row in x_values if row]
    y_flat = [row[0] for row in y_values if row]

    x_data = np.array(_to_float_list(x_flat), dtype=float)
    y_data = np.array(_to_float_list(y_flat), dtype=float)

    # ========== LOGIC (unchanged) ==========
    def fourPL(x, a, b, c, d):
        return d + (a - d) / (1.0 + (x / c)**b)

    a_guess = float(np.min(y_data))
    d_guess = float(np.max(y_data))
    c_guess = float(np.median(x_data))
    b_guess = -1.0
    initial_guesses = [a_guess, b_guess, c_guess, d_guess]

    popt, pcov = curve_fit(
        fourPL, x_data, y_data,
        p0=initial_guesses,
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, 0, np.inf, np.inf]),
        maxfev=10000
    )

    a, b, c, d = popt
    y_pred = fourPL(x_data, *popt)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # ========== END LOGIC ==========

    # --- Optional write-back to the Sheet (U17:U21 by default) ---
    if write_back:
        m = re.match(r"^([A-Z]+)(\d+)$", out_start_cell)
        if not m:
            raise ValueError(f"Invalid OUT_START_CELL: {out_start_cell}")
        col_letters, row0 = m.group(1), int(m.group(2))
        def addr(r): return f"{col_letters}{r}"
        sheet.update(addr(row0 + 0), [[float(a)]])
        sheet.update(addr(row0 + 1), [[float(b)]])
        sheet.update(addr(row0 + 2), [[float(c)]])
        sheet.update(addr(row0 + 3), [[float(d)]])
        sheet.update(addr(row0 + 4), [[float(r_squared)]])

    # --- Emit result.json for later webhook step ---
    result = {
        "job_id": job_id,
        "status": "completed",
        "message": "4PL fit completed",
        "summary": {
            "a": float(a), "b": float(b), "c": float(c), "d": float(d),
            "r2": float(r_squared),
            "x_range": x_range, "y_range": y_range
        }
    }
    with open("result.json", "w") as f:
        json.dump(result, f)

if __name__ == "__main__":
    import os
    main()
