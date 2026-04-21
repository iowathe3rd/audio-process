from __future__ import annotations
from dagster import Definitions, load_assets_from_modules
from app.dagster import assets

all_assets = load_assets_from_modules([assets])

defs = Definitions(
    assets=all_assets,
    resources={"pipeline_settings": assets.PipelineConfigResource()},
)
