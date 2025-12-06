# Team Member Model Mapping

This document describes how team member names are mapped to weather inference models for the Hive Weather Inference API.

## Team Members

The following team members have models available for inference:

1. **KAMILE SEIDU**
2. **JAMES WEDAM ANEWENAH**
3. **ANTHONY SEDJOAH**
4. **Nana Duah**
5. **FRANKLIN HOGBA**
6. **MASHUD BAWA ABDULAI**
7. **Eric Okyere**
8. **Alexander Adade**
9. **PELEG TEYE DARKEY**

## Model Loading

### How It Works

1. **Model File Naming Convention**:
   - Model files should be named using the team member's name in lowercase with underscores
   - Format: `{name_lowercase_with_underscores}.pkl`
   - Example: `kamile_seidu.pkl`, `james_wedam_anewenah.pkl`, etc.

2. **Model Location**:
   - Models should be placed in the `models/` directory
   - Path: `models/{team_member_name}.pkl`

3. **Automatic Loading**:
   - On application startup, the system attempts to load a model file for each team member
   - If a model file exists, it is loaded
   - If a model file doesn't exist, a dummy model is used for that team member

4. **Model Selection**:
   - Users can select any team member's model from the dropdown in the UI
   - Each team member's model can be used independently for inference

## Example Model Files

```
models/
├── kamile_seidu.pkl
├── james_wedam_anewenah.pkl
├── anthony_sedjoah.pkl
├── nana_duah.pkl
├── franklin_hogba.pkl
├── mashud_bawa_abdulai.pkl
├── eric_okyere.pkl
├── alexander_adade.pkl
└── peleg_teye_darkey.pkl
```

## Using Team Member Models

### Via API

```bash
# Single prediction using a team member's model
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [25.0, 60.0, 1013.25, 10.0, 10.0, 30.0],
    "model_name": "KAMILE SEIDU"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[25.0, 60.0, 1013.25, 10.0, 10.0, 30.0], [15.0, 80.0, 1005.0, 25.0, 5.0, 90.0]],
    "model_name": "JAMES WEDAM ANEWENAH"
  }'
```

### Via Web UI

1. Navigate to http://localhost:8000/home
2. Select a team member from the "Team Member / Model" dropdown
3. Enter your features
4. Click "Predict" to get the inference result

## Configuration

Team member names are configured in `app/config.py`:

```python
TEAM_MEMBERS: List[str] = [
    "KAMILE SEIDU",
    "JAMES WEDAM ANEWENAH",
    # ... etc
]
```

## Notes

- Model names are case-sensitive and must match exactly as configured
- If a model file is not found, a dummy model will be used (useful for testing)
- All team member models are loaded on application startup
- The `/api/v1/models` endpoint returns all available model names
