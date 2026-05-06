# Fish Predictor

A simple Flask web app to predict fish species or characteristics.

## Structure

- `app.py`: Main Flask app
- `model_utils.py`: Model loading and prediction utilities
- `models/`: Directory for ML models
- `templates/`: HTML templates
- `static/`: CSS and JS files
- `requirements.txt`: Python dependencies

## Usage

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the app:
   ```bash
   python app.py
   ```

