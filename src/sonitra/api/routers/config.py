from fastapi import APIRouter, HTTPException, Request, status

from sonitra.config import ConfigError, PipelineConfig

router = APIRouter()


@router.get("/config")
async def get_config(request: Request) -> dict:
    cfg = request.app.state.config
    return cfg.model_dump(mode="json")


@router.put("/config")
async def put_config(payload: dict, request: Request) -> dict:
    try:
        cfg = PipelineConfig.model_validate(payload)
    except ConfigError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    request.app.state.config = cfg
    return cfg.model_dump(mode="json")
