<!--In progress-->
# REST API

You can control Sonitra over the web using its REST API. A REST API lets your programs send requests to Sonitra over HTTP. Start the server with `sonitra serve --port 8000` or with Python code:

```python
import uvicorn
from sonitra.api.app import create_app

uvicorn.run(create_app(), host="0.0.0.0", port=8000)
```

In this example, `create_app` builds the web app. `uvicorn` is the web server that runs it. Port 8000 is the network port where you reach it. A job means one render task that Sonitra runs for you. SSE means server-sent events, a way for the server to send live updates to you.

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
