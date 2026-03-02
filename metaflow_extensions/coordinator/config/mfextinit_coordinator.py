def get_pinned_conda_libs(python_version, datastore_type):
    return {
        "httpx": ">=0.24",
        "fastapi": ">=0.100",
        "uvicorn": ">=0.23",
    }
