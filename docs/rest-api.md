# REST API

Start the server with `sonitra serve --port 8000` or programmatically:

```python
import uvicorn
from sonitra.api.app import create_app

uvicorn.run(create_app(), host="0.0.0.0", port=8000)
```

| Endpoint | Description |
|---|---|
| `GET /health` | Health check |
| `GET /ready` | Readiness check |
| `POST /jobs` | Create a render job |
| `GET /jobs` | List all jobs |
| `GET /jobs/{id}` | Get job status |
| `DELETE /jobs/{id}` | Cancel/delete a job |
| `GET /config` | Get current config |
| `PUT /config` | Reload config at runtime |
| `GET /status/{job_id}/stream` | SSE status stream |

---
[← Back to README](../README.md)
