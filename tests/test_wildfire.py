def test_health_check(client):
    """Verifies that the core infrastructure health router responds with 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200


def test_wildfire_status_endpoint(client):
    """Ensures the system configuration and model state parameters are reachable."""
    response = client.get("/wildfire/status")
    assert response.status_code == 200
    json_data = response.json()
    assert "weights_exist" in json_data
    assert "data_processing_active" in json_data


def test_predict_endpoint_success_flow(client):
    """Validates full inference pipeline using a typical severe hazard input payload."""
    payload = {
        "X": 7,
        "Y": 5,
        "month": "aug",
        "day": "sat",
        "FFMC": 94.2,
        "DMC": 120.5,
        "DC": 602.4,
        "ISI": 15.8,
        "temp": 38.4,
        "RH": 18.0,
        "wind": 28.5,
        "rain": 0.0,
    }
    response = client.post("/wildfire/predict", json=payload)
    assert response.status_code == 200

    json_data = response.json()
    assert json_data["status"] == "success"
    assert "fire_prediction" in json_data
    assert isinstance(json_data["fire_probability"], float)


def test_predict_endpoint_invalid_calendar_strings(client):
    """Ensures that invalid month names are filtered before reaching PyTorch layers."""
    payload = {
        "X": 7,
        "Y": 5,
        "month": "invalid",
        "day": "sat",
        "FFMC": 90.0,
        "DMC": 40.0,
        "DC": 400.0,
        "ISI": 10.0,
        "temp": 30.0,
        "RH": 30.0,
        "wind": 15.0,
        "rain": 0.0,
    }
    response = client.post("/wildfire/predict", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid date parameters."


def test_predict_endpoint_coordinate_out_of_bounds(client):
    """Verifies that Pydantic input field boundary constraints work as intended."""
    payload = {
        "X": 99,
        "Y": 5,
        "month": "aug",
        "day": "sat",
        "FFMC": 90.0,
        "DMC": 40.0,
        "DC": 400.0,
        "ISI": 10.0,
        "temp": 30.0,
        "RH": 30.0,
        "wind": 15.0,
        "rain": 0.0,
    }
    response = client.post("/wildfire/predict", json=payload)
    assert response.status_code == 422
