import click
import yaml
from jinja2 import Template


@click.command()
@click.option("-f", "--fileconfig", default="metaflow_config.yaml", type=str)
@click.option("-c", "--config", default=None, type=dict)
def create_metaflow_boilerplate(fileconfig: str, config: dict = None):
    if config is not None:
        info = config.copy()
    else:
        info = yaml.safe_load(open(fileconfig, "r"))

    with open("metaflow.template", "r") as f:
        boiler_plate_template = Template(f.read())

    boiler_plate = boiler_plate_template.render(**info)

    with open(info["filename"], "w") as f:
        f.write(boiler_plate)


if __name__ == "__main__":
    create_metaflow_boilerplate()
