import os
import time
import json
import csv
import datetime
import logging
import pathlib
import re
import math

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import httpx
import google.generativeai as genai
import json5  # tolerant JSON loader
from jsonschema import validate, ValidationError

# Celery for job processing
from celery import Celery
from celery.result import AsyncResult

# Add to imports at top
import holidays

# -----------------------------
# Setup & Configuration
# -----------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
logger.info("Initializing FastAPI server...")

# Configure CORS as needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global API keys (raise error if missing)
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not POLYGON_API_KEY or not GEMINI_API_KEY:
    raise ValueError("Missing required API keys in environment variables")
logger.info("Environment variables loaded.")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)
logger.info("Gemini API configured.")

# -----------------------------
# Celery Configuration
# -----------------------------
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND")

celery_app = Celery(
    "tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND
)
celery_app.conf.update(
    task_track_started=True,
    task_serializer='json',
    result_serializer='json',
    accept_content=['json']
)
celery_app.autodiscover_tasks(['main'])
logger.info(f"Celery configured with broker: {CELERY_BROKER_URL} and backend: {CELERY_RESULT_BACKEND}")

# -----------------------------
# Helper Functions
# -----------------------------
def update_progress(task, progress: int, status: str, logs: list = None):
    """Update task progress."""
    task.update_state(
        state="PROGRESS",
        meta={
            "progress": progress,
            "status": status,
            "logs": logs or []
        }
    )

def get_market_holidays(date_obj: datetime.date, api_key: str) -> list:
    """Generate US market holidays using holidays library"""
    year = date_obj.year
    
    # Get all observed US holidays
    federal_holidays = holidays.US(
        years=[year-1, year, year+1],
        observed=True
    )
    
    # Add stock market specific holidays not in federal list
    extra_holidays = {
        datetime.date(year-1, 4, 7): "Good Friday",
        datetime.date(year, 3, 29): "Good Friday",
        datetime.date(year+1, 4, 18): "Good Friday",
    }
    
    # Combine dates from both sources
    return sorted(
        list(federal_holidays.keys()) + 
        list(extra_holidays.keys())
    )

def is_trading_day(date_obj: datetime.date, holidays: list) -> bool:
    """Check if date is a trading day (weekday and not holiday)"""
    return date_obj.weekday() < 5 and date_obj not in holidays

def adjust_valuation_date(date_obj: datetime.date, api_key: str) -> datetime.date:
    """Find nearest trading day accounting for weekends AND holidays"""
    holidays_list = get_market_holidays(date_obj, api_key)
    
    # Check in order: -1, -2, -3, -4, -5, -6, -7, +1, +2, +3, +4, +5, +6, +7
    for delta in [0, -1, -2, -3, -4, -5, -6, -7, 1, 2, 3, 4, 5, 6, 7]:
        test_date = date_obj + datetime.timedelta(days=delta)
        if test_date.weekday() < 5 and test_date not in holidays_list:
            return test_date
            
    return date_obj  # fallback if no trading day found

def fetch_json(url: str) -> dict:
    with httpx.Client() as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()

def get_market_data(symbol: str, date_str: str, api_key: str) -> dict:
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{date_str}/{date_str}?apiKey={api_key}"
    try:
        data = fetch_json(url)
        if data.get("results"):
            result = data["results"][0]
            return {
                "high": safe_calculate(lambda: float(result["h"])),
                "low": safe_calculate(lambda: float(result["l"])),
                "closing": safe_calculate(lambda: float(result["c"])),
                "valuation_method": "Direct"
            }
        return {"high": None, "low": None, "closing": None, "valuation_method": "No Data"}
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return {"high": None, "low": None, "closing": None, "valuation_method": "Error"}

def get_market_data_simple(symbol: str, date_obj: datetime.date, api_key: str) -> dict:
    date_str = date_obj.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{date_str}/{date_str}?apiKey={api_key}"
    try:
        data = fetch_json(url)
        if data.get("results"):
            result = data["results"][0]
            return {"high": result["h"], "low": result["l"], "closing": result["c"]}
    except Exception:
        pass
    return None

def get_dividend_data(symbol: str, api_key: str) -> list:
    if not symbol or symbol.lower() == "null":
        return []
    url = f"https://api.polygon.io/v3/reference/dividends?ticker={symbol}&apiKey={api_key}"
    try:
        data = fetch_json(url)
        return data.get("results", [])
    except Exception as e:
        logger.error(f"Error fetching dividend data for {symbol}: {e}")
        return []

def compute_accrued_dividends(security: dict, date_of_death: str, api_key: str) -> float:
    symbol = security.get("ticker") or security.get("Symbol") or security.get("Ticker/Symbol Data") or security.get("cusip") or security.get("Cusip") or security.get("Ticker")
    if not symbol:
        return 0.0
    quantity = float(security.get("quantity", 0))
    dod = datetime.datetime.strptime(date_of_death, "%Y-%m-%d").date()
    dividends = get_dividend_data(symbol, api_key)
    accrued_dividends = 0.0
    for dividend in dividends:
        try:
            record_date = datetime.datetime.strptime(dividend.get("record_date"), "%Y-%m-%d").date()
            payment_date = datetime.datetime.strptime(dividend.get("pay_date"), "%Y-%m-%d").date()
        except Exception:
            continue
        if record_date <= dod < payment_date:
            accrued_dividends += float(dividend.get("cash_amount", 0)) * quantity
    logger.info(f"Accrued dividends for {symbol}: {accrued_dividends}")
    return accrued_dividends

def safe_calculate(calculation, default=None):
    """Enhanced calculation with numeric validation"""
    try:
        result = calculation()
        if isinstance(result, (int, float)) and not math.isnan(result):
            return result
        return default
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        return default

def compute_security_fmv(security: dict, date_of_death: str, api_key: str) -> tuple[float, list]:
    logs = []
    market_data = {}  # Initialize empty dict
    valuation_date = None
    logger.info(f"Processing security: {security.get('description')}")
    
    # Handle quantity - convert to float or default to 0
    try:
        quantity = float(security.get("quantity", 0) or 0)  # Handle None case
    except (ValueError, TypeError):
        quantity = 0
        logger.warning("Invalid quantity; using 0.")
    
    try:
        dod = datetime.datetime.strptime(date_of_death, "%Y-%m-%d").date()
    except Exception as e:
        logger.error(f"Invalid date format: {e}")
        return 0.0, logs

    symbol = (
        security.get("ticker") 
        or security.get("Symbol") 
        or security.get("Ticker/Symbol Data")
        or security.get("cusip")
        or security.get("Cusip")
        or security.get("Ticker")  # Add additional fallback
    )
    avg_price = None
    valuation_method = None

    if not symbol:
        logger.info("No ticker symbol found; using fallback from statement values.")
        try:
            market_value = str(security.get("market_value", "0")).replace("$", "").replace(",", "").strip()
            avg_price = safe_calculate(
                lambda: float(market_value) / quantity if quantity else 0,
                default=0
            )
            valuation_method = "Market Value Fallback"
        except Exception:
            avg_price = 0
            valuation_method = "No Data"
    else:
        # New logic: Check if date is a holiday and find nearest trading days
        valuation_date = adjust_valuation_date(dod, api_key)
        if valuation_date != dod:  # Adjusted for holiday/weekend
            # Check if adjustment was for holiday vs weekend
            if dod.weekday() < 5:  # Was a weekday but market closed (holiday)
                prev_day = valuation_date  # Already the nearest prior trading day
                # Get first trading day AFTER holiday
                next_day = adjust_valuation_date(dod + datetime.timedelta(days=1), api_key)
            else:  # Weekend
                if dod.weekday() == 5:  # Saturday
                    prev_day = valuation_date - datetime.timedelta(days=1)
                    next_day = valuation_date + datetime.timedelta(days=2)
                else:  # Sunday
                    prev_day = valuation_date - datetime.timedelta(days=2)
                    next_day = valuation_date + datetime.timedelta(days=1)

            prev_data = get_market_data_simple(symbol, prev_day, api_key)
            next_data = get_market_data_simple(symbol, next_day, api_key)
            
            if prev_data and next_data:
                # Use same proration logic for both holidays and weekends
                avg_prev = (prev_data["high"] + prev_data["low"]) / 2
                avg_next = (next_data["high"] + next_data["low"]) / 2
                avg_price = safe_calculate(lambda: ((avg_prev + avg_next) / 2))
                security["fmv_high"] = max(prev_data["high"], next_data["high"])
                security["fmv_low"] = min(prev_data["low"], next_data["low"])
                valuation_method = "Holiday/Weekend Prorated"
            else:
                # Fallback to valuation date data
                market_data = get_market_data(symbol, valuation_date.strftime("%Y-%m-%d"), api_key)
                security["fmv_high"] = market_data.get("high")
                security["fmv_low"] = market_data.get("low")
                avg_price = safe_calculate(lambda: (market_data.get("high", 0) + market_data.get("low", 0)) / 2) or market_data.get("closing")
                valuation_method = market_data.get("valuation_method", "Direct")
        else:
            market_data = get_market_data(symbol, valuation_date.strftime("%Y-%m-%d"), api_key)
            avg_price = safe_calculate(lambda: (market_data.get("high", 0) + market_data.get("low", 0)) / 2) or market_data.get("closing")
            valuation_method = market_data.get("valuation_method", "Direct")

    # Calculate FMV using quantity and average price
    fmv = safe_calculate(lambda: avg_price * quantity if avg_price is not None else 0)
    accrued_dividends = safe_calculate(lambda: compute_accrued_dividends(security, date_of_death, api_key), 0.0)
    total_fmv = safe_calculate(lambda: fmv + accrued_dividends, 0.0)

    security["Valuation Method"] = valuation_method
    logs.append(f"Calculated FMV for {symbol or 'Unknown Security'}: {total_fmv or 0:.2f}")
    logs.append(f"Final FMV calculation: {total_fmv or 0:.2f}")

    # Only set these if they haven't been set by weekend logic
    if "fmv_high" not in security:
        security["fmv_high"] = market_data.get("high", "")
    if "fmv_low" not in security:
        security["fmv_low"] = market_data.get("low", "")
    security["fmv_avg"] = avg_price  # This should always be set

    # Add these lines after calculating market data
    adjusted_valuation_date = adjust_valuation_date(dod, api_key)
    security["valuation_date_used"] = adjusted_valuation_date.strftime("%Y-%m-%d") if adjusted_valuation_date else ""
    security["date_of_death"] = dod.strftime("%Y-%m-%d")  # Always preserve original DoD
    security["accrued_dividends"] = accrued_dividends

    return round(total_fmv, 2), logs

def add_fmv_valuation(data: dict, date_of_death: str, api_key: str) -> tuple[dict, list]:
    all_logs = []
    positions_key = next((k for k in data.keys() if k.strip().lower().rstrip(':') == "account positions"), None)
    if not positions_key:
        return data, ["Warning: No account positions found"]
    
    positions = data[positions_key]
    total_positions = len(positions)
    for idx, security in enumerate(positions, 1):
        progress = 60 + int(30 * (idx/total_positions))  # 60-90% range for FMV
        fmv, security_logs = compute_security_fmv(security, date_of_death, api_key)
        security["Calculated FMV"] = fmv
        all_logs.extend(security_logs)  # Add FMV logs to main logs
        all_logs.append(f"PROGRESS:{progress}% - Processed {security.get('description')}")
    return data, all_logs

def save_to_csv(data: dict, csv_file: str):
    logger.info(f"Writing CSV file: {csv_file}")
    positions_key = next((k for k in data.keys() if k.strip().lower().rstrip(':') == "account positions"), None)
    if not positions_key or not isinstance(data.get(positions_key), list):
        logger.warning("No account positions found for CSV output.")
        return

    fieldnames = [
        "Ticker", "Cusip", "Description", "Quantity", 
        "Statement Market Value", "Statement Date", "Date Acquired",
        "Cost Basis", "FMV High", "FMV Low", "FMV Avg", 
        "FMV Value", "FMV Valuation Method", "Dividends Accrued",
        "FMV Valuation Date", "Valuation Date Used"
    ]
    
    rows = []
    for security in data[positions_key]:
        row = {
            "Ticker": security.get("ticker"),
            "Cusip": security.get("cusip"),
            "Description": security.get("description"),
            "Quantity": security.get("quantity"),
            "Statement Market Value": security.get("market_value"),
            "Statement Date": data.get("Account Information", {}).get("statement_date"),
            "Date Acquired": security.get("date_acquired"),
            "Cost Basis": security.get("cost_basis"),
            "FMV High": security.get("fmv_high", ""),
            "FMV Low": security.get("fmv_low", ""),
            "FMV Avg": security.get("fmv_avg", ""),
            "FMV Value": security.get("Calculated FMV"),
            "FMV Valuation Method": security.get("Valuation Method"),
            "Dividends Accrued": security.get("accrued_dividends", 0),
            "FMV Valuation Date": security.get("date_of_death", ""),  # Use original DoD for display
            "Valuation Date Used": security.get("valuation_date_used", "")  # Show adjusted date
        }
        rows.append(row)

    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"CSV file saved to {csv_file}")

def save_output(data: dict, output_file: str) -> str:
    """Save processed data to appropriate format in downloads directory."""
    download_dir = pathlib.Path("downloads")
    download_dir.mkdir(exist_ok=True)
    
    # Extract components for filename
    account_info = data.get("Account Information", {})
    financial_institution = account_info.get("financial_institution", "unknown").lower().replace(" ", "_")
    form_name = account_info.get("form_name", "unknown").lower().replace(" ", "_")
    statement_date = account_info.get("statement_date", "unknown_date").replace("/", "_")
    
    # Get date of death from first security's metadata (assuming it's consistent across all)
    date_of_death = "unknown_date"
    if data.get("Account Positions"):
        first_security = data["Account Positions"][0]
        if "date_of_death" in first_security:
            date_of_death = first_security["date_of_death"].replace("/", "_")
    
    # Construct filename
    base_filename = f"{financial_institution}_{form_name}_{statement_date}_{date_of_death}"
    extension = pathlib.Path(output_file).suffix
    
    # Clean filename of any special characters and spaces
    base_filename = re.sub(r'[^\w\-_.]', '_', base_filename)
    filename = f"{base_filename}{extension}"
    
    output_path = download_dir / filename
    
    if output_file.lower().endswith('.csv'):
        save_to_csv(data, str(output_path))
    else:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    logger.info(f"Output saved to {output_path}")
    return filename  # Return just the filename, not the full path

def upload_to_gemini(path: str, mime_type: str = None):
    file = genai.upload_file(path, mime_type=mime_type)
    logger.info(f"Uploaded file '{file.display_name}' with URI: {file.uri}")
    return file

def wait_for_files_active(files: list):
    logger.info("Waiting for Gemini file processing to complete...")
    for file in files:
        while file.state.name == "PROCESSING":
            logger.info("Gemini file still processing...")
            time.sleep(10)
            file = genai.get_file(file.name)
        if file.state.name != "ACTIVE":
            raise Exception(f"File {file.name} failed to process")
    logger.info("Gemini files are active.")

def log_and_get_status(message: str, progress: int = None) -> str:
    logger.info(message)
    status_message = f"status:{message}\n"
    if progress is not None:
        status_message += f"progress:{progress}\n"
    return status_message

def safe_json5_load(filename: str):
    """Attempt to load JSON using json5; return None on failure."""
    try:
        with open(filename, "r") as f:
            return json5.load(f)
    except Exception as e:
        logger.error(f"safe_json5_load failed: {e}")
        return None

def clean_json_response(text: str) -> str:
    """Clean and validate JSON response from Gemini."""
    # Remove any potential markdown formatting including variants with/without json
    text = re.sub(r'```(json)?', '', text, flags=re.IGNORECASE)
    
    # Remove any remaining backticks (individual or pairs)
    text = re.sub(r'`+', '', text)
    
    # Remove any potential comments
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    
    # Ensure property names are properly quoted
    def quote_properties(match):
        return f'"{match.group(1)}":'
    
    text = re.sub(r'(\w+):\s*', quote_properties, text)
    
    # Remove any trailing commas before closing braces/brackets
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    
    # Remove extra characters that might come through
    text = text.strip().strip('*').strip()
    
    # Add explicit check for remaining backticks
    if '`' in text:
        logger.warning("Found remaining backticks in cleaned JSON")
        text = text.replace('`', '')
    
    return text

# -----------------------------
# Celery Task (No Parsing of Gemini Response)
# -----------------------------
@celery_app.task(bind=True, name='main.process_file_task')
def process_file_task(self, file_path: str, date_of_death: str, output_type: str):
    """
    Process flow:
      1. Upload PDF to Gemini and wait for processing.
      2. Start a Gemini chat session with a prompt.
      3. Write Gemini's raw response to a temporary file (without parsing).
      4. Log the raw response (also write it to a separate file for inspection).
      5. If CSV/Excel output is requested, try to load and convert the file.
         On failure, fall back to returning the raw file.
    """
    logs = []
    try:
        update_progress(self, 0, "Starting processing")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        logs.append(log_and_get_status(f"Processing file: {os.path.basename(file_path)}"))
        self.update_state(state="PROGRESS", meta={"progress": 5, "status": "File processing started"})

        # Upload file to Gemini and wait for processing.
        logs.append(log_and_get_status("Uploading file to Gemini"))
        self.update_state(state="PROGRESS", meta={"progress": 20, "status": "Uploading to Gemini"})
        gemini_file = upload_to_gemini(file_path, "application/pdf")
        logs.append(log_and_get_status(f"File uploaded to Gemini: {gemini_file.uri}"))

        logs.append(log_and_get_status("Waiting for Gemini processing"))
        self.update_state(state="PROGRESS", meta={"progress": 30, "status": "Waiting for Gemini processing"})
        wait_for_files_active([gemini_file])
        logs.append(log_and_get_status("Gemini processing complete"))

        # Start Gemini chat session with prompt.
        self.update_state(state="PROGRESS", meta={"progress": 40, "status": "Initializing Gemini model"})
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            # Request plain text so Gemini doesn't force-parse JSON:
            "response_mime_type": "text/plain",
        }
        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=generation_config,
        )
        self.update_state(state="PROGRESS", meta={"progress": 45, "status": "Starting Gemini chat session"})
        chat_session = model.start_chat(history=[{
            "role": "user",
            "parts": [(
                "STRICT JSON FORMATTING REQUIRED. Parse the financial document and output EXCLUSIVELY as JSON with:\n"
                "- Account Information: Form_name, Account_holder, Account_number, Financial_institution, Statement Date\n"
                "- Account Positions: Ticker, Cusip, Description, Date_acquired, Quantity, Cost_basis, Market_value\n"
                "RULES:\n"
                "1. Use ONLY double quotes\n"
                "2. No trailing commas\n"
                "3. No JSON wrapping text\n"
                "4. Null for missing values\n"
                "5. Numeric values without symbols (e.g. '5000' not '$5,000')\n"
                "6. Date format: YYYY-MM-DD\n"
                "7. For 1099 forms use YEAR-12-31 as the statement date\n"
                "8. For monthly statements use last day of month as the statement date\n"
                "9. Please make sure that ticker is not confused with cusip\n"
                "10. No string values for numbers such as cost basis, quantity, market value, etc.\n"
                "EXAMPLE:\n"
                "{\n"
                "  \"Account Information\": {\n"
                "    \"form_name\": \"...\",\n"
                "    \"account_holder\": \"...\"\n"
                "    \"statement_date\": \"2023-12-31\",\n"
                "  },\n"
                "  \"Account Positions\": [\n"
                "    {\n"
                "      \"ticker\": \"AAPL\",\n"
                "      \"quantity\": 100\n"
                "    }\n"
                "  ]\n"
                "}"
            )]
        }])
        prompt_text = "Parse the attached document and output the entire document as JSON."
        self.update_state(state="PROGRESS", meta={"progress": 50, "status": "Sending prompt to AI"})
        response = chat_session.send_message([
            {"text": prompt_text},
            {"file_data": {"mime_type": "application/pdf", "file_uri": gemini_file.uri}}
        ])
        if not response.text:
            raise RuntimeError("Empty response from Gemini")
        self.update_state(state="PROGRESS", meta={"progress": 55, "status": "Received response from AI"})

        # Add enhanced logging for Gemini response
        raw_response = response.text
        print(f"\n\n=== RAW GEMINI RESPONSE ===\n{raw_response}\n=== END RESPONSE ===\n\n")  # Log full response
        cleaned_response = clean_json_response(raw_response)
        
        # Add validation step
        SCHEMA = {
            "type": "object",
            "properties": {
                "Account Information": {
                    "type": "object",
                    "properties": {
                        "form_name": {"type": ["string", "null"]},
                        "account_holder": {"type": ["string", "null"]},
                        "account_number": {"type": ["string", "null"]},
                        "financial_institution": {"type": ["string", "null"]},
                        "statement_date": {"type": ["string", "null"], "format": "date"}
                    },
                    "required": ["form_name", "account_holder", "statement_date"]
                },
                "Account Positions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": ["string", "null"]},
                            "cusip": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]},
                            "date_acquired": {"type": ["string", "null"]},
                            "quantity": {"type": ["number", "string", "null"]},
                            "cost_basis": {"type": ["number", "null"]},
                            "market_value": {"type": ["number", "string", "null"]},
                            "fmv_high": {"type": ["number", "null"]},
                            "fmv_low": {"type": ["number", "null"]},
                            "fmv_avg": {"type": ["number", "null"]},
                            "valuation_date": {"type": ["string", "null"]},
                            "accrued_dividends": {"type": ["number", "null"]},
                            "Calculated FMV": {"type": ["number", "null"]},
                            "Valuation Method": {"type": ["string", "null"]}
                        },
                        "required": ["description", "quantity"]
                    }
                }
            },
            "required": ["Account Information", "Account Positions"]
        }

        try:
            # Handle double-encoded JSON strings
            if isinstance(cleaned_response, str) and cleaned_response.startswith(('"{', "'{")):
                cleaned_response = json5.loads(cleaned_response)
            raw_data = json5.loads(cleaned_response) if isinstance(cleaned_response, str) else cleaned_response
            validate(instance=raw_data, schema=SCHEMA)
            
            # Add FMV calculations with logging
            try:
                raw_data, fmv_logs = add_fmv_valuation(raw_data, date_of_death, POLYGON_API_KEY)
                logs.extend(fmv_logs)  # Add FMV logs to main logs
                update_progress(self, 60, "FMV calculations complete", logs)
            except Exception as fmv_error:
                logger.error(f"FMV calculation failed: {fmv_error}")
                logs.append(f"FMV Error: {str(fmv_error)}")
                raise RuntimeError(f"FMV calculation error: {str(fmv_error)}")

            # Save output
            output_filename = save_output(raw_data, f"{date_of_death}.{output_type.lower()}")
            logs.append(log_and_get_status(f"Output saved to {output_filename}"))

        except (ValueError, ValidationError, json.JSONDecodeError) as e:
            logger.error(f"JSON validation failed. Raw response: {raw_response}\nCleaned response: {cleaned_response}\nError: {str(e)}")
            logs.append(f"JSON Error: {str(e)}")
            raise RuntimeError(f"Invalid JSON structure: {str(e)}")

        return {
            "state": "SUCCESS",
            "progress": 100,
            "status": "JSON generation complete",
            "download_file": output_filename,
            "logs": logs
        }

    except Exception as e:
        error_msg = f"Task failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file after error: {file_path}")
            except Exception as cleanup_err:
                logger.error(f"Error cleaning up file: {cleanup_err}")
        exc_info = {
            'exc_type': type(e).__name__,
            'exc_message': str(e).split('\n')[0],
            'exc_module': type(e).__module__
        }
        raise Exception(json.dumps(exc_info))

# -----------------------------
# FastAPI Endpoints
# -----------------------------
@app.post("/api/process")
def submit_process(
    file: UploadFile = File(...),
    dateOfDeath: str = Form(...),
    outputType: str = Form(...),
):
    logger.info(f"Received processing request: file={file.filename}, dateOfDeath={dateOfDeath}, outputType={outputType}")
    try:
        contents = file.file.read()
        tmp_upload = f"tmp_{int(time.time())}.pdf"
        with open(tmp_upload, "wb") as f_out:
            f_out.write(contents)
        logger.info(f"Saved temporary file: {tmp_upload}")
        task = process_file_task.apply_async(args=[tmp_upload, dateOfDeath, outputType])
        logger.info(f"Created Celery task with ID: {task.id}")
        return {"job_id": task.id}
    except Exception as e:
        logger.error(f"Error in submit_process endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/job_status/{job_id}")
def job_status(job_id: str):
    try:
        result = AsyncResult(job_id, app=celery_app)
        state = result.state
        if state == "PENDING":
            return {"job_id": job_id, "state": "PENDING", "progress": 0, "status": "Job pending..."}
        if state == "PROGRESS":
            info = result.info or {}
            return {
                "job_id": job_id,
                "state": "PROGRESS",
                "progress": info.get("progress", 0),
                "status": info.get("status", "Processing..."),
                "download_file": info.get("download_file", "")
            }
        if result.successful():
            # Extract filename from the nested result
            task_result = result.get()
            filename = task_result.get("download_file", "") if isinstance(task_result, dict) else ""
            return {"job_id": job_id, "state": "SUCCESS", "progress": 100, "status": "Complete", "download_file": filename}
        if result.failed():
            exc = result.result
            return {"job_id": job_id, "state": "FAILURE", "progress": 0, "status": f"Failed: {str(exc)}"}
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}", exc_info=True)
        return {"job_id": job_id, "state": "ERROR", "progress": 0, "status": f"Error checking status: {str(e)}"}

@app.get("/api/download/{filename:str}")
async def download_file(filename: str):
    """Handle file downloads with proper filename validation and content type"""
    try:
        # Ensure the filename is properly sanitized
        safe_filename = re.sub(r'[^\w\-_.]', '_', filename)
        file_path = pathlib.Path("downloads") / safe_filename
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine content type based on file extension
        content_type = "text/csv" if filename.lower().endswith('.csv') else "application/json"
        
        # Set content disposition to force download with original filename
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"',
            'Content-Type': content_type
        }
        
        return FileResponse(
            path=file_path,
            headers=headers,
            filename=filename
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail="Download failed")

if __name__ == "__main__":
    logger.info("Starting FastAPI server on 127.0.0.1:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
